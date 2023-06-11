import { getData, postData } from './apiCallWrappers';

export const postRegister = async (username: string, email: string, password: string) => {
    const params = {
        email: email,
    };
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);

    const response = await postData('/auth/register', formData, params, 'multipart/form-data');
    return response;
};

export const postLogin = async (email: string, password: string) => {
    const formData = new FormData();
    formData.append('username', email);
    formData.append('password', password);

    const response = await postData('/auth/login', formData, null, 'multipart/form-data');

    return response;
};

export const postDataset = async (dataset: File, userID: number) => {
    const formData = new FormData();
    // console.log(dataset.name);
    formData.append('dataset', new Blob([dataset], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }), dataset.name);
    return postData(`/data-mining/${userID}/upload-dataset`, formData, null, 'multipart/form-data');
};
