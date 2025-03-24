Based on the provided pin and power domain information, generate design validation rules to ensure proper isolation control and reliable power domain management.
Consider situations when isolation attributes (isolated and isolated_enable) are used, and specify appropriate conditions for power domain relationships between pins and their associated power sources.
Please clearly state each rule and briefly explain its physical reasoning.

Each rule must follow this format:
Rule N: [Short and clear condition in natural language]
Example: [A minimal JSON-style snippet to illustrate this rule]



Based on the provided pin and power domain information, generate design validation rules to ensure proper isolation control and reliable power domain management.
Consider situations when isolation attributes (isolated and isolated_enable) are used.

Each rule must follow this format:
Rule N: [Short and clear condition in natural language]
Example: [A minimal JSON-style snippet to illustrate this rule]


Based on the provided pin and power domain information, generate design validation rules to ensure proper isolation control and reliable power domain management.
Consider situations when isolation attributes (isolated and isolated_enable and pg type) are used.

Each rule must follow this format:
Rule N: [Short and clear condition in natural language]
Example: [A minimal JSON-style snippet to illustrate this rule]


You are an expert in power management design. Below is an example definition of a component, including various power domains (e.g., primary_power, internal_power, and primary_ground) and related pin definitions:

[Insert your example definition here.]

I am currently developing a rule-checking system designed to automatically validate the correctness and appropriateness of attributes assigned to these pins according to general power management practices.

Recently, I introduced a new pin attribute called isolated. However, the exact scenarios or rules under which this attribute should be set are still unclear.

Based on the provided component example, please analyze and infer:

Under what general circumstances or conditions would it be appropriate or necessary to set the new isolated attribute on a pin?

Please ensure your inferred rules are generalized enough to apply broadly (i.e., not specific to any pin names), and use general attributes such as related_power_pin, the types of power domains (pg_type), pin directions, and similar properties.


