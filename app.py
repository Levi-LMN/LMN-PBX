# app.py
"""
Main application with integrated ARI agent
"""

import os
import logging
import atexit
import signal
import sys
import threading
import asyncio
from flask import Flask, redirect, url_for
from flask_login import LoginManager

# Import configuration
from config import config

# Import models and database
from models import db, User

# Import blueprints
from blueprints.auth.routes import auth_bp
from blueprints.calls.routes import calls_bp
from blueprints.admin.routes import admin_bp

# Import ARI agent
from services.ari_agent import ARIAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global agent instance
_ari_agent = None
_agent_thread = None
_shutdown_initiated = False


def create_app(config_name='default'):
    """
    Application factory pattern.
    Creates and configures the Flask application with integrated ARI agent.
    """
    app = Flask(__name__)

    # Load configuration
    app.config.from_object(config[config_name])

    # Initialize extensions
    db.init_app(app)

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    # Register blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(calls_bp)
    app.register_blueprint(admin_bp)

    # Add custom Jinja filters
    @app.template_filter('from_json')
    def from_json_filter(value):
        """Parse JSON string to Python object."""
        if not value:
            return []
        try:
            import json
            return json.loads(value)
        except:
            return []

    # Root route
    @app.route('/')
    def index():
        return redirect(url_for('admin.dashboard'))

    # Initialize database
    with app.app_context():
        # Create database tables if they don't exist
        try:
            db.create_all()
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
            raise

        # Create default admin user if none exists
        try:
            if not User.query.filter_by(username=app.config['ADMIN_USERNAME']).first():
                admin = User(
                    username=app.config['ADMIN_USERNAME'],
                    email='admin@example.com',
                    role='admin'
                )
                admin.set_password(app.config['ADMIN_PASSWORD'])
                db.session.add(admin)
                db.session.commit()
                logger.info(f"Created default admin user: {app.config['ADMIN_USERNAME']}")
            else:
                logger.info("Admin user already exists")
        except Exception as e:
            logger.error(f"Error creating admin user: {e}")
            db.session.rollback()

        # Import Department model
        from models import Department

        # Create default departments if none exist
        try:
            if Department.query.count() == 0:
                default_departments = [
                    {'name': 'Sales', 'extension': '1000', 'priority': 10,
                     'description': 'Sales and new customer inquiries'},
                    {'name': 'Support', 'extension': '1005', 'priority': 5,
                     'description': 'Technical support and customer service'},
                    {'name': 'Claims', 'extension': '1002', 'priority': 8,
                     'description': 'Insurance claims processing'},
                    {'name': 'Billing', 'extension': '1003', 'priority': 7,
                     'description': 'Billing and payment inquiries'},
                ]

                for dept_data in default_departments:
                    dept = Department(**dept_data)
                    db.session.add(dept)

                db.session.commit()
                logger.info("Created default departments")
            else:
                logger.info(f"Found {Department.query.count()} existing departments")
        except Exception as e:
            logger.error(f"Error creating departments: {e}")
            db.session.rollback()

        logger.info("Application initialized successfully")

    return app


def start_ari_agent(app):
    """Start the ARI agent in a background thread"""
    global _ari_agent, _agent_thread

    logger.info("=" * 60)
    logger.info("STARTING ARI AGENT")
    logger.info("=" * 60)

    try:
        # Create agent with app config
        _ari_agent = ARIAgent(app.config)

        # Run agent in separate thread with its own event loop
        def run_agent():
            """Run the agent in a new event loop"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_ari_agent.start())
            except Exception as e:
                logger.error(f"Agent error: {e}")
            finally:
                loop.close()

        _agent_thread = threading.Thread(target=run_agent, daemon=True, name="ARI-Agent")
        _agent_thread.start()

        logger.info("✅ ARI Agent started successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to start ARI agent: {e}")
        logger.warning("Application will run without call handling")


def cleanup_on_shutdown(app):
    """Clean up resources when shutting down."""
    global _shutdown_initiated, _ari_agent

    if _shutdown_initiated:
        return

    _shutdown_initiated = True
    logger.info("=" * 60)
    logger.info("SHUTDOWN SEQUENCE INITIATED")
    logger.info("=" * 60)

    try:
        # Stop ARI agent
        if _ari_agent:
            logger.info("Stopping ARI agent...")
            try:
                # Signal the agent to stop
                asyncio.run(_ari_agent.stop())
                logger.info("ARI agent stopped")
            except Exception as e:
                logger.debug(f"Error stopping agent: {e}")

        logger.info("=" * 60)
        logger.info("SHUTDOWN COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
    finally:
        import time
        time.sleep(0.1)


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {signum}")

    cleanup_on_shutdown(None)

    logger.info("Forcing process exit...")
    os._exit(0)


if __name__ == '__main__':
    # Get configuration from environment
    env = os.getenv('FLASK_ENV', 'development')

    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create application
    app = create_app(env)

    # Start ARI agent in background
    start_ari_agent(app)

    # Make agent available to routes
    app.ari_agent = _ari_agent

    # Register cleanup for normal exit
    atexit.register(lambda: cleanup_on_shutdown(app))

    try:
        # Run development server
        logger.info("Starting Flask application...")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 60)

        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Disable debug to prevent reloader issues
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
    finally:
        logger.info("Flask server stopped")
        cleanup_on_shutdown(app)