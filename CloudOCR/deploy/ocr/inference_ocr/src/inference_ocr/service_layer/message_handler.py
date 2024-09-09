import logging

from ..utils.exceptions import (
    TritonServerError,
    MessageHandlerError,
)


class MessageHandler:
    def __init__(self, text_detector, text_recognizer):

        self.log = logging.getLogger("Message_handler")

        self.text_detector = text_detector
        self.text_recognizer = text_recognizer

        self.log.info("Initialised message_handler")

    def __call__(self, image):

        self.log.info("Handling message")
        try:
    
            # Process image
            couple_cropped_images = self.text_detector(image) #there are two images per crop.

            final_results = []
            for couple_crop in couple_cropped_images:
                raw_output_1, confidence_1 = self.text_recognizer(couple_crop["crop1"])
                raw_output_2, confidence_2 = self.text_recognizer(couple_crop["crop2"])
                final_results.append({"raw_output_1": raw_output_1, "raw_output_2": raw_output_2, "confidence_1": confidence_1, "confidence_2": confidence_2, "coordinates": couple_crop["coordinates"]})
                # final_results.append({"raw_output_1": raw_output_1, "raw_output_2": raw_output_2, "confidence_1": confidence_1, "confidence_2": confidence_2, "coordinates"})
                # final_results.append({"raw_output_1": raw_output_1, "raw_output_2": raw_output_2, "confidence_1": confidence_1, "confidence_2": confidence_2})
        
        except (TritonServerError, Exception) as error:
            raise MessageHandlerError(error)
        self.log.info("Finished handling image")

        return final_results
