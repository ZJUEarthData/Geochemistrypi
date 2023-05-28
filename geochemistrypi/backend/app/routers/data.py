import pandas as pd
from fastapi import APIRouter, UploadFile

router = APIRouter(
    prefix="/data",
    tags=["data"],
    responses={404: {"description": "Not found"}},
)

fake_data = None


@router.post("/upload")
async def upload_data(data: UploadFile):
    # print(data.filename)
    # print(data.content_type)
    contents = await data.read()
    df = pd.read_excel(contents)
    print(df)
    processed_data = df.to_json(orient="records")
    # Process the uploaded file as needed
    # e.g., save it to a specific location or perform data processing
    # return {"message": "Data uploaded successfully"}
    return processed_data


# router.get("/get")


# @router.post("/upload")
# async def upload_data(data: bytes = File(...)):
#     # print(data)
#     print(pd.read_excel(data))
#     # Process the uploaded file as needed
#     # e.g., save it to a specific location or perform data processing
#     return {'message': 'Data uploaded successfully'}
