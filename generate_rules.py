from liberty.parser import parse_liberty
from llm_request import request_internal_llm
from langchain.schema import HumanMessage

def generate_initial_prompt(cell_group):
    """Generate initial consistency rules from the NLDM Lib file."""
    prompt_template = f'''
    Please generate consistency rules based on the NLDM Lib File. The rules are as follows:

    Rule 1: In the pg_pin, if a pin contains the switch_function attribute, it must have the switch_pin: true property.
    Example: pg_pin(VDDAI) {{ switch_function: "SD"; }}
    
    Please generate additional consistency rules and present them in human language.
    {cell_group}
    '''
    return prompt_template

def evaluate_rules(llm_response, cell_group):
    """Evaluate, refine, and prioritize the generated rules."""
    prompt_template = f'''
    You are an experienced NLDM Lib CAD engineer, and your task is to verify the accuracy of each rule and provide modification suggestions.

    Below are my generated rules:
    {llm_response}

    And the corresponding Lib file content:
    {cell_group}

    The Lib file content is correct. Please carefully review each rule for accuracy and provide your assessment. 

    **Output Format**
    Rule X: Rule Description  
    Example:  
    Interpretation: Explain the underlying physical meaning in a way that junior engineers can understand.  
    Recommendation:  

    Please prioritize the rules based on accuracy and keep only the **top 10 most important rules**.
    '''
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
    return reAnswer_prompt_template

def refine_rules(cell_group, max_iteration=2):
    """Iteratively refine and format the consistency rules."""
    
    # Step 1: Generate initial rules
    print("Generating initial rules...")
    generate_initial_prompt_text = generate_initial_prompt(cell_group)
    messages = [HumanMessage(content=generate_initial_prompt_text)]
    rules = request_internal_llm(messages)
    print("Initial Rules Generated:\n", rules)

    for i in range(max_iteration):
        print(f"\nIteration {i+1}: Evaluating and refining rules...")
        
        # Step 2: Validate & filter rules
        evaluate_rules_prompt_text = evaluate_rules(rules, cell_group)
        messages = [HumanMessage(content=evaluate_rules_prompt_text)]
        refined_rules = request_internal_llm(messages)
        print("Refined Rules:\n", refined_rules)

        # Step 3: Format the rules
        reAnswer_prompt_text = re_answer(refined_rules)
        messages = [HumanMessage(content=reAnswer_prompt_text)]
        formatted_rules = request_internal_llm(messages)
        print("Formatted Rules:\n", formatted_rules)

        # Check for convergence (if refined rules ≈ previous rules, stop early)
        if formatted_rules.strip() == rules.strip():
            print("Rules have converged, stopping iteration early.")
            break
        
        # Update rules for next iteration
        rules = formatted_rules

    return formatted_rules

### 🔹 Read and parse the .lib file
with open("test.lib", "r") as f:
    lib_content = f.read()
parsed_lib = parse_liberty(lib_content)
cell_group = str(parsed_lib)

### 🔹 Run the iterative refinement process (2 iterations)
final_rules = refine_rules(cell_group, max_iteration=2)

### 🔹 Print final output
print("\nFinal Refined Rules:\n", final_rules)
