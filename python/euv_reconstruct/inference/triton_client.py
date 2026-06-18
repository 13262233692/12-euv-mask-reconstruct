from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    voxel_grid: np.ndarray
    inference_time_ms: float
    model_name: str
    model_version: str
    status: str = "success"
    error_message: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.voxel_grid.shape

    @property
    def height_field(self) -> np.ndarray:
        if self.voxel_grid.ndim == 3:
            return np.max(self.voxel_grid, axis=0)
        return self.voxel_grid

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "inference_time_ms": self.inference_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "voxel_shape": list(self.voxel_grid.shape),
            "error_message": self.error_message,
        }


class TritonVoxelInference:
    def __init__(
        self,
        server_url: str = "localhost:8001",
        model_name: str = "voxel3d_cnn",
        model_version: str = "1",
        input_name: str = "input",
        output_name: str = "output",
        timeout: float = 30.0,
        use_ssl: bool = False,
    ):
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.output_name = output_name
        self.timeout = timeout
        self.use_ssl = use_ssl

        self._client = None
        self._client_ready = False
        self._loop = None

    def _get_client(self):
        if self._client is None:
            import tritonclient.grpc as grpcclient
            import tritonclient.utils as triton_utils

            ssl_options = None
            if self.use_ssl:
                ssl_options = grpcclient.ssl_channel_options()

            try:
                self._client = grpcclient.InferenceServerClient(
                    url=self.server_url,
                    ssl=ssl_options is not None,
                    ssl_options=ssl_options,
                    verbose=False,
                )
                self._triton_utils = triton_utils
                self._client_ready = True
            except Exception as e:
                logger.error(f"Failed to create Triton client: {e}")
                raise
        return self._client

    def is_server_ready(self) -> bool:
        try:
            client = self._get_client()
            return client.is_server_ready()
        except Exception as e:
            logger.warning(f"Server readiness check failed: {e}")
            return False

    def is_model_ready(self) -> bool:
        try:
            client = self._get_client()
            return client.is_model_ready(self.model_name, self.model_version)
        except Exception as e:
            logger.warning(f"Model readiness check failed: {e}")
            return False

    def get_model_metadata(self) -> Dict[str, Any]:
        try:
            client = self._get_client()
            metadata = client.get_model_metadata(
                self.model_name, self.model_version
            )
            return {
                "name": metadata.name,
                "versions": list(metadata.versions),
                "platform": metadata.platform,
                "inputs": [
                    {
                        "name": inp.name,
                        "datatype": inp.datatype,
                        "shape": list(inp.shape),
                    }
                    for inp in metadata.inputs
                ],
                "outputs": [
                    {
                        "name": out.name,
                        "datatype": out.datatype,
                        "shape": list(out.shape),
                    }
                    for out in metadata.outputs
                ],
            }
        except Exception as e:
            logger.error(f"Failed to get model metadata: {e}")
            raise

    def _prepare_input_tensor(
        self,
        input_data: np.ndarray,
    ) -> "grpcclient.InferInput":
        import tritonclient.grpc as grpcclient

        if input_data.ndim == 3:
            input_data = input_data[np.newaxis, ...]
        if input_data.ndim == 4:
            input_data = input_data[np.newaxis, ...]

        input_data = input_data.astype(np.float32)

        infer_input = grpcclient.InferInput(
            self.input_name,
            input_data.shape,
            "FP32",
        )
        infer_input.set_data_from_numpy(input_data)

        return infer_input

    def _prepare_output_tensor(self) -> "grpcclient.InferRequestedOutput":
        import tritonclient.grpc as grpcclient

        infer_output = grpcclient.InferRequestedOutput(
            self.output_name,
            class_count=0,
        )
        return infer_output

    def infer(
        self,
        input_tensor: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        start_time = time.time()

        try:
            client = self._get_client()

            infer_input = self._prepare_input_tensor(input_tensor)
            infer_output = self._prepare_output_tensor()

            params = parameters or {}

            response = client.infer(
                model_name=self.model_name,
                inputs=[infer_input],
                outputs=[infer_output],
                model_version=self.model_version,
                parameters=params,
                client_timeout=self.timeout,
            )

            output_data = response.as_numpy(self.output_name)

            if output_data.ndim == 5:
                output_data = output_data[0, 0]
            elif output_data.ndim == 4:
                output_data = output_data[0]

            inference_time = (time.time() - start_time) * 1000

            return InferenceResult(
                voxel_grid=output_data,
                inference_time_ms=inference_time,
                model_name=self.model_name,
                model_version=self.model_version,
                status="success",
            )

        except Exception as e:
            inference_time = (time.time() - start_time) * 1000
            logger.error(f"Inference failed: {e}")
            return InferenceResult(
                voxel_grid=np.array([]),
                inference_time_ms=inference_time,
                model_name=self.model_name,
                model_version=self.model_version,
                status="failed",
                error_message=str(e),
            )

    async def infer_async(
        self,
        input_tensor: np.ndarray,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.infer(input_tensor, parameters),
        )

    def infer_batch(
        self,
        input_tensors: List[np.ndarray],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[InferenceResult]:
        results = []
        for tensor in input_tensors:
            result = self.infer(tensor, parameters)
            results.append(result)
        return results

    async def infer_batch_async(
        self,
        input_tensors: List[np.ndarray],
        parameters: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 4,
    ) -> List[InferenceResult]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_infer(tensor):
            async with semaphore:
                return await self.infer_async(tensor, parameters)

        tasks = [bounded_infer(tensor) for tensor in input_tensors]
        results = await asyncio.gather(*tasks)
        return list(results)

    def infer_stream(
        self,
        input_tensor: np.ndarray,
        callback,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            client = self._get_client()

            infer_input = self._prepare_input_tensor(input_tensor)
            infer_output = self._prepare_output_tensor()

            client.start_stream(callback=callback)

            client.async_stream_infer(
                model_name=self.model_name,
                inputs=[infer_input],
                outputs=[infer_output],
                model_version=self.model_version,
                parameters=parameters or {},
            )

            client.stop_stream()

        except Exception as e:
            logger.error(f"Stream inference failed: {e}")
            raise

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._client_ready = False

    def __enter__(self):
        self._get_client()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
