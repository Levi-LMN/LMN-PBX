# blueprints/auth/routes.py
"""
Authentication blueprint for user login, logout, and management.
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from datetime import datetime
from models import db, User

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if current_user.is_authenticated:
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember', False)

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            if not user.is_active:
                flash('Your account has been deactivated. Please contact an administrator.', 'error')
                return redirect(url_for('auth.login'))

            login_user(user, remember=bool(remember))
            user.last_login = datetime.utcnow()
            db.session.commit()

            # Redirect to next page or dashboard
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('admin.dashboard'))
        else:
            flash('Invalid username or password', 'error')

    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    """Handle user logout."""
    logout_user()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('auth.login'))


@auth_bp.route('/users')
@login_required
def list_users():
    """List all users (admin only)."""
    if not current_user.has_permission('admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    users = User.query.all()
    return render_template('admin/users.html', users=users)


@auth_bp.route('/users/create', methods=['GET', 'POST'])
@login_required
def create_user():
    """Create new user (admin only)."""
    if not current_user.has_permission('admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role', 'viewer')

        # Validate
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('admin/user_form.html')

        # Create user
        user = User(username=username, email=email, role=role)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash(f'User {username} created successfully', 'success')
        return redirect(url_for('auth.list_users'))

    return render_template('admin/user_form.html')


@auth_bp.route('/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    """Edit existing user (admin only)."""
    if not current_user.has_permission('admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        user.email = request.form.get('email')
        user.role = request.form.get('role')
        user.is_active = bool(request.form.get('is_active'))

        # Update password if provided
        new_password = request.form.get('password')
        if new_password:
            user.set_password(new_password)

        db.session.commit()
        flash(f'User {user.username} updated successfully', 'success')
        return redirect(url_for('auth.list_users'))

    return render_template('admin/user_form.html', user=user)


@auth_bp.route('/users/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete user (admin only)."""
    if not current_user.has_permission('admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    user = User.query.get_or_404(user_id)

    # Prevent deleting yourself
    if user.id == current_user.id:
        flash('You cannot delete your own account', 'error')
        return redirect(url_for('auth.list_users'))

    username = user.username
    db.session.delete(user)
    db.session.commit()

    flash(f'User {username} deleted successfully', 'success')
    return redirect(url_for('auth.list_users'))