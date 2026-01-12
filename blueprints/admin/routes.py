# blueprints/admin/routes.py
"""
Admin blueprint - Complete with all routes
Fixed with real file system access (no SSH)
"""

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
from sqlalchemy import func, desc
from datetime import datetime, timedelta
import json
import time

from models import db, Call, CallTranscript, CallIntent, Department, RoutingRule, KnowledgeBase

admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


# ============================================================================
# Dashboard and Analytics
# ============================================================================

@admin_bp.route('/')
@admin_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with statistics and recent activity."""
    # Get statistics
    total_calls = Call.query.count()
    active_calls = Call.query.filter_by(status='active').count()
    escalated_calls = Call.query.filter_by(escalated=True).count()

    # Calls in last 24 hours
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_calls = Call.query.filter(Call.started_at >= yesterday).count()

    # Average call duration
    avg_duration = db.session.query(func.avg(Call.duration_seconds)).scalar() or 0

    # Intent distribution
    intent_stats = db.session.query(
        CallIntent.intent_type,
        func.count(CallIntent.id)
    ).group_by(CallIntent.intent_type).all()

    # Recent calls (last 10)
    recent_call_list = Call.query.order_by(desc(Call.started_at)).limit(10).all()

    # Get call volume for last 7 days (for chart)
    chart_labels = []
    chart_data = []

    if total_calls > 0:
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        daily_calls = db.session.query(
            func.date(Call.started_at).label('date'),
            func.count(Call.id).label('count')
        ).filter(
            Call.started_at >= seven_days_ago
        ).group_by(
            func.date(Call.started_at)
        ).order_by('date').all()

        for i in range(6, -1, -1):
            date = (datetime.utcnow() - timedelta(days=i)).date()
            chart_labels.append(date.strftime('%a'))

            count = 0
            for daily in daily_calls:
                if str(daily.date) == str(date):
                    count = daily.count
                    break
            chart_data.append(count)
    else:
        for i in range(6, -1, -1):
            date = (datetime.utcnow() - timedelta(days=i)).date()
            chart_labels.append(date.strftime('%a'))
            chart_data.append(0)

    return render_template('admin/dashboard.html',
                           total_calls=total_calls,
                           active_calls=active_calls,
                           escalated_calls=escalated_calls,
                           recent_calls=recent_calls,
                           avg_duration=int(avg_duration),
                           intent_stats=intent_stats,
                           recent_call_list=recent_call_list,
                           chart_labels=chart_labels,
                           chart_data=chart_data)


@admin_bp.route('/api/system-status')
@login_required
def system_status():
    """
    API endpoint for real system status - NO SSH CHECK
    """
    status = {
        'timestamp': datetime.utcnow().isoformat(),
        'services': {}
    }

    # Check ARI Agent
    ari_agent = getattr(current_app, 'ari_agent', None)
    if ari_agent and ari_agent.running:
        status['services']['ari'] = {
            'name': 'ARI Agent',
            'status': 'running',
            'details': {
                'active_calls': len(ari_agent.active_calls),
                'total_calls': ari_agent.total_calls,
                'server': f"{ari_agent.ari_base}"
            },
            'icon': 'fa-phone',
            'color': 'success'
        }
    else:
        status['services']['ari'] = {
            'name': 'ARI Agent',
            'status': 'not_running',
            'details': {'message': 'ARI agent not initialized or stopped'},
            'icon': 'fa-phone',
            'color': 'danger'
        }

    # Check AI Service
    if ari_agent and ari_agent.ai_client:
        status['services']['ai'] = {
            'name': 'AI Assistant',
            'status': 'connected',
            'details': {
                'provider': 'Azure OpenAI',
                'model': ari_agent.azure_openai_deployment,
                'endpoint': ari_agent.azure_openai_endpoint[:50] + '...' if len(ari_agent.azure_openai_endpoint) > 50 else ari_agent.azure_openai_endpoint
            },
            'icon': 'fa-brain',
            'color': 'success'
        }
    else:
        status['services']['ai'] = {
            'name': 'AI Assistant',
            'status': 'not_configured',
            'details': {'message': 'Check AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT in .env'},
            'icon': 'fa-brain',
            'color': 'danger'
        }

    # Check Speech Service
    if ari_agent and ari_agent.transcriber:
        status['services']['speech'] = {
            'name': 'Speech Services',
            'status': 'connected',
            'details': {
                'provider': 'Azure Cognitive Services',
                'region': ari_agent.azure_speech_region
            },
            'icon': 'fa-microphone',
            'color': 'success'
        }
    else:
        status['services']['speech'] = {
            'name': 'Speech Services',
            'status': 'not_configured',
            'details': {'message': 'Check AZURE_SPEECH_KEY in .env'},
            'icon': 'fa-microphone',
            'color': 'danger'
        }

    # Check File System Access (replaces SSH)
    if ari_agent and ari_agent.file_access:
        if ari_agent.file_access.can_write:
            status['services']['filesystem'] = {
                'name': 'Audio File System',
                'status': 'connected',
                'details': {
                    'directory': ari_agent.asterisk_sounds_dir,
                    'access': 'sudo' if ari_agent.file_access.use_sudo else 'direct'
                },
                'icon': 'fa-folder',
                'color': 'success'
            }
        else:
            status['services']['filesystem'] = {
                'name': 'Audio File System',
                'status': 'read_only',
                'details': {
                    'message': 'No write permission to /var/lib/asterisk/sounds/custom',
                    'directory': ari_agent.asterisk_sounds_dir
                },
                'icon': 'fa-folder',
                'color': 'warning'
            }
    else:
        status['services']['filesystem'] = {
            'name': 'Audio File System',
            'status': 'not_configured',
            'details': {'message': 'File system access not initialized'},
            'icon': 'fa-folder',
            'color': 'warning'
        }

    # Overall system health
    all_services = list(status['services'].values())
    connected_count = sum(1 for s in all_services if s['status'] in ['connected', 'running'])
    total_count = len(all_services)

    status['overall'] = {
        'health': 'healthy' if connected_count == total_count else 'partial' if connected_count > 0 else 'down',
        'connected_services': connected_count,
        'total_services': total_count,
        'percentage': round((connected_count / total_count) * 100) if total_count > 0 else 0
    }

    return jsonify(status)


@admin_bp.route('/api/active-calls')
@login_required
def active_calls_status():
    """Get list of currently active calls with real-time data"""
    ari_agent = getattr(current_app, 'ari_agent', None)

    if not ari_agent or not hasattr(ari_agent, 'active_calls'):
        return jsonify({'active_calls': [], 'count': 0})

    active_calls_list = []
    current_time = time.time()

    for call in ari_agent.active_calls:
        try:
            # Calculate duration from channel creation time
            creation_time = call.channel.json.get('creationtime', current_time)
            duration = int(current_time - creation_time)

            call_info = {
                'call_id': call.id[:12],
                'caller_number': call.channel.json.get('caller', {}).get('number', 'Unknown'),
                'duration': duration,
                'interactions': call.turn_count
            }
            active_calls_list.append(call_info)
        except Exception as e:
            # Skip calls that error
            continue

    return jsonify({
        'active_calls': active_calls_list,
        'count': len(active_calls_list),
        'timestamp': datetime.utcnow().isoformat()
    })


@admin_bp.route('/analytics')
@login_required
def analytics():
    """Comprehensive analytics and statistics page."""
    total_calls = Call.query.count()

    if total_calls == 0:
        # Return empty data if no calls
        return render_template('admin/analytics.html',
                               total_calls=0, calls_growth=0, avg_duration_min=0, avg_duration_sec=0,
                               resolution_rate=0, calls_today=0, avg_interactions=0, escalation_rate=0,
                               success_rate=0, peak_hour='N/A', peak_hour_calls=0,
                               volume_labels=[], volume_data=[], intent_labels=[], intent_data=[],
                               status_labels=[], status_data=[], hourly_labels=[], hourly_data=[],
                               dept_labels=[], dept_data=[], response_time_data=[0, 0, 0, 0, 0],
                               top_callers=[])

    # Growth calculation
    now = datetime.utcnow()
    this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    last_month_start = (this_month_start - timedelta(days=1)).replace(day=1)

    calls_this_month = Call.query.filter(Call.started_at >= this_month_start).count()
    calls_last_month = Call.query.filter(
        Call.started_at >= last_month_start,
        Call.started_at < this_month_start
    ).count()

    calls_growth = round((calls_this_month - calls_last_month) / calls_last_month * 100) if calls_last_month > 0 else 0

    # Average duration
    avg_duration = db.session.query(func.avg(Call.duration_seconds)).filter(
        Call.duration_seconds.isnot(None)
    ).scalar() or 0
    avg_duration_min = int(avg_duration // 60)
    avg_duration_sec = int(avg_duration % 60)

    # Resolution rate
    escalated_calls = Call.query.filter_by(escalated=True).count()
    resolution_rate = round((total_calls - escalated_calls) / total_calls * 100) if total_calls > 0 else 0

    # Calls today
    yesterday = datetime.utcnow() - timedelta(days=1)
    calls_today = Call.query.filter(Call.started_at >= yesterday).count()

    # Average interactions
    avg_interactions = round(db.session.query(func.avg(Call.total_interactions)).scalar() or 0, 1)

    # Escalation and success rates
    escalation_rate = round(escalated_calls / total_calls * 100) if total_calls > 0 else 0
    success_rate = 100 - escalation_rate

    # Call Volume Trend (30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    daily_calls = db.session.query(
        func.date(Call.started_at).label('date'),
        func.count(Call.id).label('count')
    ).filter(
        Call.started_at >= thirty_days_ago
    ).group_by(func.date(Call.started_at)).order_by('date').all()

    volume_labels = []
    volume_data = []
    for i in range(29, -1, -1):
        date = (datetime.utcnow() - timedelta(days=i)).date()
        volume_labels.append(date.strftime('%m/%d'))
        count = 0
        for daily in daily_calls:
            if str(daily.date) == str(date):
                count = daily.count
                break
        volume_data.append(count)

    # Intent Distribution
    intents = db.session.query(
        CallIntent.intent_type,
        func.count(CallIntent.id)
    ).group_by(CallIntent.intent_type).all()

    intent_labels = [i[0].capitalize() for i in intents]
    intent_data = [i[1] for i in intents]

    # Status Breakdown
    statuses = db.session.query(
        Call.status,
        func.count(Call.id)
    ).group_by(Call.status).all()

    status_labels = [s[0].capitalize() for s in statuses]
    status_data = [s[1] for s in statuses]

    # Hourly Pattern
    hourly_calls = db.session.query(
        func.strftime('%H', Call.started_at).label('hour'),
        func.count(Call.id).label('count')
    ).group_by('hour').all()

    hourly_labels = [f"{i:02d}:00" for i in range(24)]
    hourly_data = []
    for i in range(24):
        count = 0
        for hour_data in hourly_calls:
            if hour_data.hour and int(hour_data.hour) == i:
                count = hour_data.count
                break
        hourly_data.append(count)

    # Peak hour
    peak_index = hourly_data.index(max(hourly_data)) if hourly_data else 0
    peak_hour = hourly_labels[peak_index] if hourly_labels else 'N/A'
    peak_hour_calls = max(hourly_data) if hourly_data else 0

    # Department Escalations
    dept_escalations = db.session.query(
        Department.name,
        func.count(Call.id)
    ).join(
        Call,
        Call.escalated_to_department_id == Department.id
    ).filter(
        Call.escalated == True
    ).group_by(Department.name).all()

    dept_labels = [d[0] for d in dept_escalations] if dept_escalations else []
    dept_data = [d[1] for d in dept_escalations] if dept_escalations else []

    if not dept_labels:
        total_escalated = Call.query.filter_by(escalated=True).count()
        if total_escalated > 0:
            dept_labels = ['Unassigned']
            dept_data = [total_escalated]

    # Response Time Distribution
    response_times = db.session.query(CallTranscript.ai_response_time_ms).filter(
        CallTranscript.ai_response_time_ms.isnot(None),
        CallTranscript.speaker == 'assistant'
    ).all()

    response_time_data = [0, 0, 0, 0, 0]

    if response_times:
        for rt in response_times:
            ms = rt[0]
            if ms < 500:
                response_time_data[0] += 1
            elif ms < 1000:
                response_time_data[1] += 1
            elif ms < 1500:
                response_time_data[2] += 1
            elif ms < 2000:
                response_time_data[3] += 1
            else:
                response_time_data[4] += 1

    # Top Callers
    top_callers_data = db.session.query(
        Call.caller_number,
        func.count(Call.id).label('count'),
        func.avg(Call.duration_seconds).label('avg_duration')
    ).group_by(Call.caller_number).order_by(func.count(Call.id).desc()).limit(10).all()

    top_callers = [{
        'phone': c[0],
        'count': c[1],
        'avg_duration': round(c[2]) if c[2] else 0
    } for c in top_callers_data]

    return render_template('admin/analytics.html',
                           total_calls=total_calls,
                           calls_growth=calls_growth,
                           avg_duration_min=avg_duration_min,
                           avg_duration_sec=avg_duration_sec,
                           resolution_rate=resolution_rate,
                           calls_today=calls_today,
                           avg_interactions=avg_interactions,
                           escalation_rate=escalation_rate,
                           success_rate=success_rate,
                           peak_hour=peak_hour,
                           peak_hour_calls=peak_hour_calls,
                           volume_labels=volume_labels,
                           volume_data=volume_data,
                           intent_labels=intent_labels,
                           intent_data=intent_data,
                           status_labels=status_labels,
                           status_data=status_data,
                           hourly_labels=hourly_labels,
                           hourly_data=hourly_data,
                           dept_labels=dept_labels,
                           dept_data=dept_data,
                           response_time_data=response_time_data,
                           top_callers=top_callers)


# ============================================================================
# Call Logs and Details
# ============================================================================

@admin_bp.route('/calls')
@login_required
def call_logs():
    """View all call logs with filtering and pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = 50

    # Filters
    status_filter = request.args.get('status')
    escalated_filter = request.args.get('escalated')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')

    # Build query
    query = Call.query

    if status_filter:
        query = query.filter_by(status=status_filter)

    if escalated_filter:
        query = query.filter_by(escalated=(escalated_filter.lower() == 'true'))

    if date_from:
        try:
            date_from_obj = datetime.strptime(date_from, '%Y-%m-%d')
            query = query.filter(Call.started_at >= date_from_obj)
        except ValueError:
            pass

    if date_to:
        try:
            date_to_obj = datetime.strptime(date_to, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(Call.started_at < date_to_obj)
        except ValueError:
            pass

    # Order by most recent
    query = query.order_by(desc(Call.started_at))

    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    calls = pagination.items

    return render_template('admin/call_logs.html',
                           calls=calls,
                           pagination=pagination,
                           status_filter=status_filter,
                           escalated_filter=escalated_filter,
                           date_from=date_from,
                           date_to=date_to)


@admin_bp.route('/calls/<int:call_id>')
@login_required
def call_detail(call_id):
    """View detailed information about a specific call."""
    call = Call.query.get_or_404(call_id)

    # Get transcripts in chronological order
    transcripts = CallTranscript.query.filter_by(
        call_id=call.id
    ).order_by(CallTranscript.timestamp).all()

    # Get detected intents
    intents = CallIntent.query.filter_by(
        call_id=call.id
    ).order_by(CallIntent.detected_at).all()

    return render_template('admin/call_detail.html',
                           call=call,
                           transcripts=transcripts,
                           intents=intents)


@admin_bp.route('/calls/<int:call_id>/transcript.json')
@login_required
def call_transcript_json(call_id):
    """Get call transcript as JSON for export."""
    call = Call.query.get_or_404(call_id)
    transcripts = CallTranscript.query.filter_by(
        call_id=call.id
    ).order_by(CallTranscript.timestamp).all()

    data = {
        'call_id': call.call_id,
        'caller_number': call.caller_number,
        'started_at': call.started_at.isoformat(),
        'ended_at': call.ended_at.isoformat() if call.ended_at else None,
        'transcript': [
            {
                'timestamp': t.timestamp.isoformat(),
                'speaker': t.speaker,
                'text': t.text,
                'confidence': t.confidence
            }
            for t in transcripts
        ]
    }

    return jsonify(data)


# ============================================================================
# Department Management
# ============================================================================

@admin_bp.route('/departments')
@login_required
def departments():
    """View and manage departments."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    depts = Department.query.all()
    return render_template('admin/departments.html', departments=depts)


@admin_bp.route('/departments/create', methods=['GET', 'POST'])
@login_required
def create_department():
    """Create new department."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        extension = request.form.get('extension')
        priority = request.form.get('priority', 0, type=int)

        if Department.query.filter_by(name=name).first():
            flash('Department with this name already exists', 'error')
            return render_template('admin/department_form.html')

        dept = Department(
            name=name,
            description=description,
            extension=extension,
            priority=priority
        )

        db.session.add(dept)
        db.session.commit()

        flash(f'Department {name} created successfully', 'success')
        return redirect(url_for('admin.departments'))

    return render_template('admin/department_form.html')


@admin_bp.route('/departments/<int:dept_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_department(dept_id):
    """Edit existing department."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    dept = Department.query.get_or_404(dept_id)

    if request.method == 'POST':
        dept.name = request.form.get('name')
        dept.description = request.form.get('description')
        dept.extension = request.form.get('extension')
        dept.priority = request.form.get('priority', 0, type=int)
        dept.is_active = bool(request.form.get('is_active'))

        db.session.commit()
        flash(f'Department {dept.name} updated successfully', 'success')
        return redirect(url_for('admin.departments'))

    return render_template('admin/department_form.html', department=dept)


@admin_bp.route('/departments/<int:dept_id>/delete', methods=['POST'])
@login_required
def delete_department(dept_id):
    """Delete department."""
    if not current_user.has_permission('admin'):
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('admin.departments'))

    dept = Department.query.get_or_404(dept_id)
    name = dept.name

    db.session.delete(dept)
    db.session.commit()

    flash(f'Department {name} deleted successfully', 'success')
    return redirect(url_for('admin.departments'))


# ============================================================================
# Routing Rules
# ============================================================================

@admin_bp.route('/routing-rules')
@login_required
def routing_rules():
    """View and manage routing rules."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    rules = RoutingRule.query.order_by(desc(RoutingRule.priority)).all()
    departments = Department.query.all()

    return render_template('admin/routing_rules.html',
                           rules=rules,
                           departments=departments)


@admin_bp.route('/routing-rules/create', methods=['GET', 'POST'])
@login_required
def create_routing_rule():
    """Create new routing rule."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        department_id = request.form.get('department_id', type=int)
        intent_type = request.form.get('intent_type')
        keywords = request.form.get('keywords', '[]')
        priority = request.form.get('priority', 0, type=int)

        rule = RoutingRule(
            department_id=department_id,
            intent_type=intent_type,
            keywords=keywords,
            priority=priority
        )

        db.session.add(rule)
        db.session.commit()

        flash('Routing rule created successfully', 'success')
        return redirect(url_for('admin.routing_rules'))

    departments = Department.query.all()
    intent_types = ['sales', 'support', 'claims', 'billing', 'escalation', 'general']

    return render_template('admin/routing_rule_form.html',
                           departments=departments,
                           intent_types=intent_types)


@admin_bp.route('/routing-rules/<int:rule_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_routing_rule(rule_id):
    """Edit existing routing rule."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    rule = RoutingRule.query.get_or_404(rule_id)

    if request.method == 'POST':
        rule.department_id = request.form.get('department_id', type=int)
        rule.intent_type = request.form.get('intent_type')
        rule.keywords = request.form.get('keywords')
        rule.priority = request.form.get('priority', 0, type=int)
        rule.is_active = bool(request.form.get('is_active'))

        db.session.commit()
        flash('Routing rule updated successfully', 'success')
        return redirect(url_for('admin.routing_rules'))

    departments = Department.query.all()
    intent_types = ['sales', 'support', 'claims', 'billing', 'escalation', 'general']

    return render_template('admin/routing_rule_form.html',
                           rule=rule,
                           departments=departments,
                           intent_types=intent_types)


@admin_bp.route('/routing-rules/<int:rule_id>/delete', methods=['POST'])
@login_required
def delete_routing_rule(rule_id):
    """Delete routing rule."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.routing_rules'))

    rule = RoutingRule.query.get_or_404(rule_id)
    db.session.delete(rule)
    db.session.commit()

    flash('Routing rule deleted successfully', 'success')
    return redirect(url_for('admin.routing_rules'))


# ============================================================================
# AI Configuration
# ============================================================================

@admin_bp.route('/ai-config', methods=['GET', 'POST'])
@login_required
def ai_config():
    """Configure AI system prompt and behavior."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        system_prompt = request.form.get('system_prompt')
        escalation_threshold = request.form.get('escalation_threshold', type=int)

        flash('AI configuration updated successfully', 'success')

        current_app.config['DEFAULT_SYSTEM_PROMPT'] = system_prompt
        current_app.config['ESCALATION_THRESHOLD'] = escalation_threshold

        return redirect(url_for('admin.ai_config'))

    system_prompt = current_app.config.get('DEFAULT_SYSTEM_PROMPT', '')
    escalation_threshold = current_app.config.get('ESCALATION_THRESHOLD', 3)

    return render_template('admin/ai_config.html',
                           system_prompt=system_prompt,
                           escalation_threshold=escalation_threshold)


# ============================================================================
# Knowledge Base
# ============================================================================

@admin_bp.route('/knowledge')
@login_required
def knowledge_base():
    """View and manage knowledge base entries."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    category_filter = request.args.get('category')

    query = KnowledgeBase.query
    if category_filter:
        query = query.filter_by(category=category_filter)

    entries = query.order_by(desc(KnowledgeBase.priority)).all()

    categories = db.session.query(KnowledgeBase.category).distinct().all()
    categories = [c[0] for c in categories]

    return render_template('admin/knowledge.html',
                           entries=entries,
                           categories=categories,
                           category_filter=category_filter)


@admin_bp.route('/knowledge/create', methods=['GET', 'POST'])
@login_required
def create_knowledge():
    """Create new knowledge base entry."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    if request.method == 'POST':
        title = request.form.get('title')
        category = request.form.get('category')
        content = request.form.get('content')
        keywords = request.form.get('keywords', '[]')
        priority = request.form.get('priority', 0, type=int)

        entry = KnowledgeBase(
            title=title,
            category=category,
            content=content,
            keywords=keywords,
            priority=priority,
            created_by=current_user.username
        )

        db.session.add(entry)
        db.session.commit()

        flash(f'Knowledge entry "{title}" created successfully', 'success')
        return redirect(url_for('admin.knowledge_base'))

    categories = ['policies', 'claims', 'billing', 'coverage', 'faq', 'procedures']
    return render_template('admin/knowledge_form.html', categories=categories)


@admin_bp.route('/knowledge/<int:entry_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_knowledge(entry_id):
    """Edit knowledge base entry."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.dashboard'))

    entry = KnowledgeBase.query.get_or_404(entry_id)

    if request.method == 'POST':
        entry.title = request.form.get('title')
        entry.category = request.form.get('category')
        entry.content = request.form.get('content')
        entry.keywords = request.form.get('keywords')
        entry.priority = request.form.get('priority', 0, type=int)
        entry.is_active = bool(request.form.get('is_active'))
        entry.version += 1

        db.session.commit()
        flash(f'Knowledge entry "{entry.title}" updated successfully', 'success')
        return redirect(url_for('admin.knowledge_base'))

    categories = ['policies', 'claims', 'billing', 'coverage', 'faq', 'procedures']
    return render_template('admin/knowledge_form.html',
                           entry=entry,
                           categories=categories)


@admin_bp.route('/knowledge/<int:entry_id>/delete', methods=['POST'])
@login_required
def delete_knowledge(entry_id):
    """Delete knowledge base entry."""
    if not current_user.has_permission('manager'):
        flash('Access denied. Manager privileges required.', 'error')
        return redirect(url_for('admin.knowledge_base'))

    entry = KnowledgeBase.query.get_or_404(entry_id)
    title = entry.title

    db.session.delete(entry)
    db.session.commit()

    flash(f'Knowledge entry "{title}" deleted successfully', 'success')
    return redirect(url_for('admin.knowledge_base'))