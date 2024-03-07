import logging

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from ..utils.exceptions import TritonServerError
from ..utils.dataclasses import ModelTritonInterface

class TritonClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        connection_attempts: int = 5,
    ):

        self.log = logging.getLogger("triton_client:")
        self.host = host
        self.port = port
        self.connection_attempts = connection_attempts

        self.triton_client = None

        self._connect(self.connection_attempts)
        
        self.log.info(f"Initialised Triton client, model")
        
    @classmethod
    def from_dict(cls, input_dict):
        return cls(
            host=input_dict["host"],
            port=input_dict["port"],
            connection_attempts=input_dict["connection_attempts"],
        )

    def __del__(self):
        self._close_connection()

    def _close_connection(self):
        """Cleanly close connection if it exists."""
        if self.triton_client and self.triton_client.is_server_ready:
            try:
                self.triton_client.close()
                self.log.info("Connection closed")
            except Exception as error:  # pylint: disable=broad-except
                self.log.error("Error while closing connection: %s", repr(error))

    def _connect(self, attempts: int = 5):
        """Attempt to connect to Triton server multiple times before giving up.

        Args:
            attempts: Number of connection attempts before giving up.
        """
        self.log.info("Attempting to connect to Triton Server")

        for i in range(attempts):
            try:
                self.log.info("Attempt %d of %d", i + 1, attempts)
                self._close_connection()
                self.triton_client = httpclient.InferenceServerClient(
                    url=f"{self.host}:{self.port}", verbose=False
                )
                if self.triton_client.is_server_ready():
                    self.log.info(
                        f"Established connection to Triton server: {self.host}:{self.port}"
                    )
                    self._running = True
                    break
            except Exception as error:
                self.log.error(
                    "Error while attempting to connect: '%s'",
                    repr(error),
                )
        else:
            raise ConnectionError(
                f"Couldn't connect to Triton after {attempts} attempts"
            )

    def __call__(self, input_data, model_triton_interface: ModelTritonInterface):

        try:
            predictions = self._infer(input_data, model_triton_interface)
            return predictions
        
        except Exception as error:
            raise TritonServerError("Failed model inference: %s", repr(error))

    def _infer(self, input_data, model_triton_interface: ModelTritonInterface):

        inputs = []

        inputs.append(
            httpclient.InferInput(
                name=model_triton_interface.model_input_name ,
                shape=input_data.shape,
                datatype=model_triton_interface.model_datatype,
            )
        )

        outputs = []
        outputs.append(httpclient.InferRequestedOutput(name=model_triton_interface.model_output_name))

        # Initialize the data per input layer
        inputs[0].set_data_from_numpy(input_data)
        try:
            # inference (http call)
            predictions = self.triton_client.infer(
                model_name=model_triton_interface.model_name, inputs=inputs, outputs=outputs
            )
            predictions= predictions.as_numpy(model_triton_interface.model_output_name)
            return predictions
        except InferenceServerException:
            raise

    def shutdown(self):
        """Graceful shutdown."""
        self.log.info("Shutting down")
        self._close_connection()
