from datetime import datetime
from fastapi.encoders import jsonable_encoder

from mlflow.gateway.config import RouteConfig
from mlflow.gateway.providers.base import BaseProvider, ProviderAdapter
from mlflow.gateway.providers.utils import send_request
from mlflow.gateway.schemas import completions

from ml_mlflow_provider.config import CustomLLMConfig


class CustomLLMAdapter(ProviderAdapter):
    @classmethod
    def completion_to_model(cls, payload, config):
        """
        將 MLflow 請求轉換為後端 LLM API 所需的格式。
        設定 model、temperature、top_p、frequency_penalty、presence_penalty 與 max_tokens，
        並將 MLflow 預設的 prompt 欄位轉換為 messages 格式。
        """
        api_request = {
            "model": "llama3.1",
            "temperature": 0.0,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 2000,
        }
        user_prompt = payload.get("prompt", "")
        api_request["messages"] = [{"role": "user", "content": user_prompt}]
        return api_request

    @classmethod
    def model_to_completions(cls, resp, config):
        """
        將後端 LLM API 的回應轉換成 MLflow Gateway / MLflow UI 所需的格式。
        為了避免夾帶多餘的 metadata，**我們將只萃取所需欄位** 重新建立一個 ResponsePayload。
        """
        # 1. 取得 model
        model_value = resp.get("model", "llama3.1")

        # 2. 取得 usage（若無則預設都是 0）
        usage_info = resp.get("usage", {})
        prompt_tokens = int(usage_info.get("prompt_tokens", 0))
        completion_tokens = int(usage_info.get("completion_tokens", 0))
        total_tokens = int(usage_info.get("total_tokens", 0))

        # 3. 嘗試從 choices 萃取內容
        choices = resp.get("choices", [])
        if isinstance(choices, list) and len(choices) > 0:
            # 預期後端第一個 choice
            choice = choices[0]

            # 預期有 message = {"content": "..."}，或直接有 text
            if "message" in choice and isinstance(choice["message"], dict):
                content = choice["message"].get("content", "")
            else:
                # 如果後端直接回傳 choice["text"]
                content = choice.get("text", "")
            finish_reason = choice.get("finish_reason", "stop")
        else:
            # 沒有 choices，就從最外層的 response 取 text，或給個空字串
            content = resp.get("response", "")
            finish_reason = "stop"

        # 4. 建立符合 MLflow Gateway 要求的 ResponsePayload
        return completions.ResponsePayload(
            object="text_completion",  # or "chat.completion" 若要顯示在聊天模式
            created=int(datetime.utcnow().timestamp()),
            model=model_value,
            choices=[
                completions.Choice(
                    index=0,
                    text=content,
                    finish_reason=finish_reason,
                )
            ],
            usage=completions.CompletionsUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        )


class CustomLLMProvider(BaseProvider):
    """
    自訂 LLM 提供者，透過後端 LLM API 處理請求，
    配置資訊（如 api_url 與 llm_api_key) 皆可在 config.yaml 中設定。
    """
    NAME = "custom_llm"
    CONFIG_TYPE = CustomLLMConfig

    def __init__(self, config: RouteConfig) -> None:
        super().__init__(config)
        self.config: CustomLLMConfig = config.model.config

    @property
    def base_url(self):
        """
        取得後端 LLM API 的 Base URL（來自 config.yaml 中的 api_url）。
        """
        return self.config.api_url

    @property
    def headers(self):
        """
        設定 HTTP 請求標頭，包含授權資訊與內容格式。
        """
        return {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "X_UID": "TESTUSE",
            "Content-Type": "application/json",
        }

    async def completions(self, payload: completions.RequestPayload) -> completions.ResponsePayload:
        """
        MLflow Gateway 接收到 /completions 請求後：
          1. 將 payload 轉換成後端 API 所需格式 (completion_to_model)
          2. 使用 send_request 非同步呼叫後端 LLM API
          3. 將 API 回應轉換成 MLflow Gateway / UI 所需的格式 (model_to_completions)
        """
        payload_dict = jsonable_encoder(payload, exclude_none=True)
        self.check_for_model_field(payload_dict)

        # 取得後端 API 所需格式
        api_request = CustomLLMAdapter.completion_to_model(payload_dict, self.config)

        # 呼叫後端 API
        resp = await send_request(
            headers=self.headers,
            base_url=self.base_url,
            path="chat",  # 後端 API 路徑
            payload=api_request,
        )

        # 轉換為 MLflow 回應格式
        return CustomLLMAdapter.model_to_completions(resp, self.config)
