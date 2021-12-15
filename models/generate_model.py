from models.generate_model_femnist import generate_model_femnist
from models.generate_model_ichar import generate_model_ichar
from models.generate_model_icsr import generate_model_icsr

def generate_model(dataset_name):
    if dataset_name == "FEMNIST":
        return generate_model_femnist()
    elif dataset_name == "ICHAR":
        return generate_model_ichar()
    elif dataset_name == "ICSR":
        return generate_model_icsr()
    else:
        return None