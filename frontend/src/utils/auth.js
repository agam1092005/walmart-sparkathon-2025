// Utility functions for authentication

export const getCookie = (name) => {
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) return parts.pop().split(';').shift();
  return null;
};

export const hasValidToken = () => {
  const token = getCookie('pb_token');
  return token !== null && token !== undefined && token !== '';
};

export const clearAuthCookies = () => {
  // Remove all cookies (esp. pb_token)
  document.cookie.split(';').forEach(cookie => {
    const eqPos = cookie.indexOf('=');
    const name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
    document.cookie = name + '=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/';
  });
}; 