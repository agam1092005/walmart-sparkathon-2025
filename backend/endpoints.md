### API Endpoints

#### Health

- **GET** `/v1/health`  
  Returns version and maintenance status.

#### Auth (defined, but not registered)

- **POST** `/signup`  
  Register a new user (org_name, email, password).
- **POST** `/signin`  
  Login with (email, password).
- **POST** `/logout`  
  Logout the current user (JWT required).
- **POST** `/forgot-password`  
  Reset password (email, new_password).
