from util import load_config, generate_model_fit, evaluate_model_on_validation

config = load_config()
model = generate_model_fit(config)  # [x^2, x, 1]

if len(model) == 2:
    print(f'CRR1: {model[2]}\n CRR2: {model[1]}\n C_d: {model[0]}')

evaluate_model_on_validation(model, config)
