import { Cookies } from 'react-cookie';

export const setUserIDCookie = (userID: string) => {
    const cookies = new Cookies();
    cookies.set('userID', userID, {
        expires: new Date(Date.now() + 1000 * 60 * 60 * 24), // // Set the cookie expiration to 1 day
        secure: true, // Set the secure flag to ensure the cookie is only sent over HTTPS
        // sameSite: "strict",  // Set the sameSite flag to 'strict' to ensure the cookie is only sent with requests to the same domain
        httpOnly: true, // Set the httpOnly flag to true to prevent client-side JavaScript from accessing the cookie
    });
};
