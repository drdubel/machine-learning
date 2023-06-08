import logging
import os
from pickle import dump, load
from secrets import token_urlsafe
from typing import Optional

from aioprometheus.asgi.middleware import MetricsMiddleware
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import Cookie, FastAPI, Response
from fastapi.staticfiles import StaticFiles
from starlette.config import Config
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse

from .data.authorized import authorized

logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="!secret")
app.add_middleware(MetricsMiddleware)

config = Config("catsndogs/data/.env")
oauth = OAuth(config)

background_task_started = False

app.mount("/static", StaticFiles(directory="./static", html=True), name="static")

CONF_URL = "https://accounts.google.com/.well-known/openid-configuration"
oauth.register(
    name="google",
    server_metadata_url=CONF_URL,
    client_kwargs={"scope": "openid email profile"},
)

with open("catsndogs/data/cookies.pickle", "rb") as cookies:
    access_cookies = load(cookies)


@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    return HTMLResponse(
        '<h1>Sio!<br>Tu nic nie ma!</h1><a href="/auto">Strona Główna</a>'
    )


@app.get("/")
async def homepage(request: Request, access_token: Optional[str] = Cookie(None)):
    user = request.session.get("user")
    if access_token in access_cookies:
        return RedirectResponse(url="/auto")
    if user:
        return HTMLResponse('<h1>Sio!</h1><a href="/login">login</a>')
    return HTMLResponse('<a href="/login">login</a>')


@app.get("/auto")
async def main(request: Request, access_token: Optional[str] = Cookie(None)):
    user = request.session.get("user")
    if user and access_token in access_cookies:
        with open(os.path.join("static", "index.html")) as fh:
            data = fh.read()
        return Response(content=data, media_type="text/html")
    return RedirectResponse(url="/")


@app.get("/login")
async def login(request: Request):
    redirect_uri = str(request.url_for("auth"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth")
async def auth(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as error:
        return HTMLResponse(f"<h1>{error.error}</h1>")
    user = token.get("userinfo")
    if user:
        request.session["user"] = dict(user)
        if user["email"] in authorized:
            access_token = token_urlsafe()
            access_cookies[access_token] = user["email"]
            with open("catsndogs/data/cookies.pickle", "wb") as cookies:
                dump(access_cookies, cookies)
            response = RedirectResponse(url="/auto")
            response.set_cookie("access_token", access_token, max_age=3600 * 24 * 14)
            return response
        else:
            return RedirectResponse(url="/")


@app.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    return RedirectResponse(url="/")


def start():
    import uvicorn

    logging.basicConfig(level=logging.DEBUG)
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    start()
