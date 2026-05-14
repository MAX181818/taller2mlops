import mlflow
from mlflow import MlflowClient

def compare_with_champion(run_id, model_name, alias, metric) -> bool:
    client = MlflowClient()
    
    # Obtenemos la métrica del modelo recién entrenado
    new_val = client.get_run(run_id).data.metrics[metric]
    
    try:
        # Buscamos el modelo champion actual
        champ = client.get_model_version_by_alias(model_name, alias)
        champ_run = client.get_run(champ.run_id).data.metrics[metric]
        
        # Si el nuevo es estrictamente mayor, se promueve
        return new_val > champ_run
    except Exception:
        # Si ocurre un error (ej. no existe ningún champion todavía), promovemos el actual por defecto
        return True

def promote_champion(run_id, model_name, alias):
    client = MlflowClient()
    # search_model_versions busca en todas las versiones, no solo la última
    versions = client.search_model_versions(f"name='{model_name}'")

    version = next((v for v in versions if v.run_id == run_id), None)
    if version is None:
        raise ValueError(f"No se encontró versión registrada para run_id={run_id}")

    # Asignar el alias 'champion' a esta nueva versión ganadora
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version.version
    )
    print(f"Champion promovido: {model_name} v{version.version}")