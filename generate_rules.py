from liberty.parser import parse_liberty
from llm_request import request_internal_llm
from langchain.schema import HumanMessage
import mlflow 
from mlflow.tracing.fluent import start_span

mlflow.set_experiment("LangChain Tracing")
mlflow.langchain.autolog()


def generate_initial_prompt(cell_group):
    """Generate initial consistency rules from the NLDM Lib file."""
    prompt_template = f'''
    Please generate consistency rules based on the NLDM Lib File. The rules are as follows:

    Rule 1: In the pg_pin, if a pin contains the switch_function attribute, it must have the switch_pin: true property.
    Example: pg_pin(VDDAI) {{ switch_function: "SD"; }}
    
    Please generate additional consistency rules and present them in human language.
    {cell_group}
    '''
    # cell_group='kelly'
    # prompt_template = f'hello {cell_group}'
    return prompt_template

def evaluate_rules(llm_response, cell_group):
    """Evaluate, refine, and prioritize the generated rules."""
    prompt_template = f'''
    You are an experienced NLDM Lib CAD engineer, and your task is to verify the accuracy of each rule and provide modification suggestions.

    Below are my generated rules:
    {llm_response}

    And the corresponding Lib file content:
    {cell_group}

    # The Lib file content is correct. Please carefully review each rule for accuracy and provide your assessment. 

    # **Output Format**
    # Rule X: Rule Description  
    # Example:  
    # Interpretation: Explain the underlying physical meaning in a way that junior engineers can understand.  
    # Recommendation:  

    # Please prioritize the rules based on accuracy and keep only the **top 10 most important rules**.
    # '''
    # cell_group='kelly'
    # prompt_template = f'hello {cell_group}'
    return prompt_template

def re_answer(reflect_response):
    """Reformat and finalize the refined rules."""
    reAnswer_prompt_template = f'''
    Based on {reflect_response}, display the final selected rules. 
    Only output the rules in the following format:
    
    Rule:
    Example:
    Interpretation:
    '''
    # cell_group='kelly'
    # prompt_template = f'hello {cell_group}'
   
    return reAnswer_prompt_template 

def refine_rules(cell_group, max_iteration=2):
    """Iteratively refine and format the consistency rules with MLflow tracking."""
    
    # 開啟 MLflow run，所有記錄會在此 run 下進行
    with mlflow.start_run() as run:
        
        
        # 記錄輸入參數，例如 cell_group 的長度
        with mlflow.start_run(nested=True):
            mlflow.log_param("cell_group_length", len(cell_group))
        # 整個流程的根 span
        with start_span(name="RefineRulesFlow") as root_span:
            root_span.set_inputs({"cell_group": cell_group})
            # Step 1: 生成初始規則
            print("Generating initial rules...")
            generate_initial_prompt_text = generate_initial_prompt(cell_group)
            
            with start_span(name="generate_initial_prompt") as sp1:
                generate_initial_prompt_text = generate_initial_prompt(cell_group)
                sp1.set_inputs({"cell_group": cell_group, "prompt_text": generate_initial_prompt_text})
                messages = [HumanMessage(content=generate_initial_prompt_text)]
                rules = request_internal_llm(messages)
                sp1.set_outputs({"rules": rules})
                mlflow.log_text(rules, artifact_file="initial_rules.txt")
                print("Initial Rules Generated:\n", rules)
            

            # 進入迭代調整規則的過程
            for i in range(max_iteration):
                print(f"\nIteration {i+1}: Evaluating and refining rules...")
                # 使用嵌套 run 來記錄當前迭代參數
                with mlflow.start_run(nested=True):
                    mlflow.log_param("iteration", i+1)

                with start_span(name="evaluate_rules") as sp2:
                    # Step 2: 驗證並過濾規則
                    evaluate_rules_prompt_text = evaluate_rules(rules, cell_group)
                    sp2.set_inputs({"previous_rules": rules, "eval_prompt_text": evaluate_rules_prompt_text})
                    messages = [HumanMessage(content=evaluate_rules_prompt_text)]
                    refined_rules = request_internal_llm(messages)
                    # 記錄此次迭代生成的規則
                    sp2.set_outputs({"refined_rules": refined_rules})
                    mlflow.log_text(refined_rules, artifact_file=f"refined_rules_iter_{i+1}.txt")
                    print("Refined Rules:\n", refined_rules)

                with start_span(name="re_answer") as sp3:
                    # Step 2: 驗證並過濾規則
                    reAnswer_prompt_text = re_answer(refined_rules)

                    # 把 prompt 記錄為inputs
                    sp3.set_inputs({"reAnswer_prompt_text":reAnswer_prompt_text})
                    messages = [HumanMessage(content=reAnswer_prompt_text)]
                    formatted_rules = request_internal_llm(messages)
                    sp3.set_outputs({"formatted_rules": formatted_rules})
                    mlflow.log_text(formatted_rules, artifact_file=f"formatted_rules_iter_{i+1}.txt")
                    print("Formatted Rules:\n", formatted_rules)
                
                # 檢查是否收斂 (格式化後的規則與前一次生成的規則一致)
                if formatted_rules.strip() == rules.strip():
                    print("Rules have converged, stopping iteration early.")
                    break
                
                # 更新規則以進行下一輪調整
                rules = formatted_rules

            root_span.set_outputs({"final_rules": formatted_rules})

        # 最後將最終結果記錄下來
        mlflow.log_text(formatted_rules, artifact_file="final_rules.txt")
        return formatted_rules


### 🔹 Read and parse the .lib file
with open('test.lib',"r") as f:
    lib_content = f.read()
parsed_lib = parse_liberty(lib_content)
cell_group = str(parsed_lib)

### 🔹 Run the iterative refinement process (2 iterations) with MLflow tracking
final_rules = refine_rules(cell_group, max_iteration=2)

### 🔹 Print final output
print("\nFinal Refined Rules:\n", final_rules)

