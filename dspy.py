import dspy
from dspy import LM
import requests

class MyCustomAPI(LM):
    def __init__(self, api_url, api_key):
        super().__init__()
        self.api_url = api_url
        self.api_key = api_key

    def basic_request(self, prompt, **kwargs):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        payload = {
            'prompt': prompt,
            'max_tokens': kwargs.get('max_tokens', 300),
            'temperature': kwargs.get('temperature', 0.1),
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        result = response.json()
        
        # 依照你API回傳的結構調整以下
        completion = result['completion']

        return completion

# 設定你自家的LLM API
my_llm = MyCustomAPI(
    api_url='https://your-company-api.com/generate',
    api_key='your-secret-key'
)

# 將DSPy設定使用你的自定義LLM
dspy.settings.configure(lm=my_llm)


# Step 1: 定義 Signature
class GenerateDesignRules(dspy.Signature):
    pins = dspy.InputField(desc="List of pin definitions.")
    pg_pins = dspy.InputField(desc="List of pg_pin definitions.")
    rules = dspy.OutputField(desc="Clearly list logical design rules inferred from pins and pg_pins.")

# Step 2: 使用內建的 ChainOfThought (Zero-shot，不給答案)
predictor = dspy.ChainOfThought(GenerateDesignRules)

# Step 3: 建立訓練範例 (至少2筆)
train_example_1 = dspy.Example(
    pins=[
    ],
    pg_pins=[
        {
        }
    ],
    rules=(
    )
).with_inputs('pins', 'pg_pins')

# 新增的第二筆範例（資料稍微變化，但規則一樣）
train_example_2 = dspy.Example(
    pins=[
        {
        },
        {
        }
    ],
    pg_pins=[
        {
        }
    ],
    rules=(
        1.
        2. 
    )
).with_inputs('pins', 'pg_pins')

trainset = [train_example_1, train_example_2]  # 至少2筆資料

# Step 4: 評估函數
def evaluate_rules(pred, gold):
    expected_rules = gold.rules.split('\n')
    return all(rule.strip() in pred.rules for rule in expected_rules)

# Step 5: Prompt Tuning using MIPRO (Zero-shot，不給答案)
teleprompter = MIPROv2(metric=evaluate_rules, verbose=True)

# 使用訓練資料優化 prompt（不會將答案顯示給prompt）
optimized_predictor = teleprompter.compile(
    predictor,
    trainset=trainset,
    num_trials=3,  # 可調整嘗試次數
    minibatch=False 
)

# 顯示最佳Prompt (透過 signature 顯示)
print("\n🔑 Optimized Zero-shot Prompt (via Signature):\n")
print(optimized_predictor) 

# Step 6: 新資料測試
test_input = {
    "pins": [
        {
            "name": "TEST_OUT",
        },
        {
        }
    ],
    "pg_pins": [
        {
        }
    ]
}

result = optimized_predictor(**test_input)

print("\n📏 Zero-shot 推理結果：")
print(result.rules)
