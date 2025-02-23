import os
from pydantic import validator
from mlflow.gateway.base_models import ConfigModel

class CustomLLMConfig(ConfigModel):
    llm_api_key: str
    api_url: str

    @validator("llm_api_key", pre=True)
    def validate_llm_api_key(cls, value):
        if value.startswith("$"):
            env_var_name = value[1:]
            return os.getenv(env_var_name, value)
        return value

    @validator("api_url", pre=True)
    def validate_api_url(cls, value):
        if value.startswith("$"):
            env_var_name = value[1:]
            return os.getenv(env_var_name, value)
        return value