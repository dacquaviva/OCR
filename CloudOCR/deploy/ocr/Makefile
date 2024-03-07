test_ocr_with_redis:
	docker compose --env-file .env -f docker-compose-redis.test.yaml up --build

# Tests
test_build:
	docker compose -f docker-compose.ci.build.yaml build inference_ocr

test_unit: test_build
	docker compose -f docker-compose.ci.test.yaml run --entrypoint=pytest inference_ocr /home/docker_user/tests/unit