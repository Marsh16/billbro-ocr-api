from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from .routes import bp as routes_bp

app = Flask(__name__)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

app.register_blueprint(routes_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)