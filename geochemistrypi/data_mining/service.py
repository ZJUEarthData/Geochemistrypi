import auth.sql_models as auth_models
from fastapi import HTTPException
from sqlalchemy.orm import Session

from .constants import MAX_UPLOADS_PER_USER
from .sql_models import Dataset


def read_all_datasets(db: Session, user_id: int):
    return db.query(Dataset).filter_by(user_id=user_id).order_by(Dataset.sequence).all()


def read_basic_datasets_info(db: Session, user_id: int):
    # Select id, name, sequence from datasets where user_id = user_id order by sequence
    basic_datasets_info_list = db.query(Dataset.id, Dataset.name, Dataset.sequence).filter_by(user_id=user_id).order_by(Dataset.sequence).all()
    # Convert the list of tuples to a list of dictionaries
    basic_datasets_info = [dict(zip(["id", "name", "sequence"], row)) for row in basic_datasets_info_list]
    return basic_datasets_info


def read_dataset(db: Session, user_id: int, dataset_id: int):
    return db.query(Dataset).filter_by(user_id=user_id, id=dataset_id).first()


def upload_dataset(db: Session, user_id: int, json_dataset: str, dataset_name: str):
    user = db.query(auth_models.User).get(user_id)
    if user.upload_count >= MAX_UPLOADS_PER_USER:
        raise HTTPException(status_code=429, detail="User has reached maximum number of uploads")

    # Calculate the new sequence number for the dataset
    new_sequence = user.upload_count + 1

    # Update the sequence numbers of existing datasets
    existing_datasets = db.query(Dataset).filter_by(user_id=user_id).order_by(Dataset.sequence)
    for dataset in existing_datasets:
        if dataset.sequence >= new_sequence:
            dataset.sequence += 1

    # Create a new Dataset instance and associate it with the user
    new_dataset = Dataset(json_data=json_dataset, sequence=new_sequence, name=dataset_name, user_id=user_id)
    db.add(new_dataset)
    # Commit the new dataset to the database
    db.commit()
    # Refresh the dataset to get the id
    db.refresh(new_dataset)

    # Update the user's upload count
    user.upload_count += 1
    db.commit()

    # Update associated diagrams
    # update_diagrams(db, new_dataset.id, json_dataset)

    return new_dataset


def remove_dataset(db: Session, user_id: int, dataset_id: int):
    dataset = db.query(Dataset).get(dataset_id)
    if not dataset:
        raise Exception("Dataset not found")

    # Delete associated diagrams
    # delete_diagrams(db, dataset_id)

    # Update the sequence numbers of existing datasets
    existing_datasets = db.query(Dataset).filter_by(user_id=user_id).order_by(Dataset.sequence)
    for other_dataset in existing_datasets:
        if other_dataset.sequence > dataset.sequence:
            other_dataset.sequence -= 1

    # Delete the dataset
    db.delete(dataset)
    db.commit()

    # Update the user's upload count
    user = db.query(auth_models.User).get(user_id)
    user.upload_count -= 1
    db.commit()

    return dataset
