# blueprints/calls/routes.py
"""
Call handling blueprint for SIP integration and call processing.
These endpoints would be called by FreePBX or the SIP client.
"""

from flask import Blueprint, request, jsonify, current_app
import logging

logger = logging.getLogger(__name__)

calls_bp = Blueprint('calls', __name__, url_prefix='/api/calls')


@calls_bp.route('/incoming', methods=['POST'])
def incoming_call():
    """
    Webhook endpoint for incoming calls from FreePBX.

    Expected payload:
    {
        "caller_number": "+15551234567",
        "called_number": "1000",
        "call_id": "unique-call-identifier"
    }
    """
    try:
        data = request.get_json()
        caller_number = data.get('caller_number')

        if not caller_number:
            return jsonify({'error': 'caller_number required'}), 400

        logger.info(f"Incoming call from {caller_number}")

        # Get call manager from app context
        call_manager = current_app.call_manager

        # Handle the call
        call_id = call_manager.handle_incoming_call(caller_number)

        return jsonify({
            'status': 'success',
            'call_id': call_id,
            'message': 'Call being processed by AI assistant'
        }), 200

    except Exception as e:
        logger.error(f"Error handling incoming call: {e}")
        return jsonify({'error': str(e)}), 500


@calls_bp.route('/audio', methods=['POST'])
def receive_audio():
    """
    Receive audio chunks from ongoing call.

    Expected: Raw audio data or base64 encoded audio
    """
    try:
        call_id = request.args.get('call_id')
        if not call_id:
            return jsonify({'error': 'call_id required'}), 400

        # Get audio data
        audio_data = request.data

        if not audio_data:
            return jsonify({'error': 'No audio data'}), 400

        call_manager = current_app.call_manager

        # In a real implementation, this would feed audio to the speech service
        # For now, we acknowledge receipt
        logger.debug(f"Received {len(audio_data)} bytes of audio for call {call_id}")

        return jsonify({'status': 'received'}), 200

    except Exception as e:
        logger.error(f"Error receiving audio: {e}")
        return jsonify({'error': str(e)}), 500


@calls_bp.route('/transcription', methods=['POST'])
def receive_transcription():
    """
    Receive transcription from external speech service.
    Alternative to built-in speech processing.

    Payload:
    {
        "call_id": "call_abc123",
        "text": "transcribed text",
        "confidence": 0.95
    }
    """
    try:
        data = request.get_json()
        call_id = data.get('call_id')
        text = data.get('text')
        confidence = data.get('confidence', 0.0)

        if not call_id or not text:
            return jsonify({'error': 'call_id and text required'}), 400

        logger.info(f"Received transcription for call {call_id}: {text}")

        call_manager = current_app.call_manager

        # Process the transcription
        call_manager._handle_user_speech(call_id, text, confidence)

        return jsonify({'status': 'processed'}), 200

    except Exception as e:
        logger.error(f"Error processing transcription: {e}")
        return jsonify({'error': str(e)}), 500


@calls_bp.route('/end', methods=['POST'])
def end_call():
    """
    End an active call.

    Payload:
    {
        "call_id": "call_abc123"
    }
    """
    try:
        data = request.get_json()
        call_id = data.get('call_id')

        if not call_id:
            return jsonify({'error': 'call_id required'}), 400

        logger.info(f"Ending call {call_id}")

        call_manager = current_app.call_manager
        call_manager.end_call_by_id(call_id)

        return jsonify({'status': 'call ended'}), 200

    except Exception as e:
        logger.error(f"Error ending call: {e}")
        return jsonify({'error': str(e)}), 500


@calls_bp.route('/status/<call_id>', methods=['GET'])
def call_status(call_id):
    """Get status of an active call."""
    try:
        call_manager = current_app.call_manager
        status = call_manager.get_call_status(call_id)

        if not status:
            return jsonify({'error': 'Call not found or inactive'}), 404

        return jsonify(status), 200

    except Exception as e:
        logger.error(f"Error getting call status: {e}")
        return jsonify({'error': str(e)}), 500


@calls_bp.route('/active', methods=['GET'])
def active_calls():
    """Get list of all active calls."""
    try:
        call_manager = current_app.call_manager
        active = call_manager.get_active_calls()

        return jsonify({
            'active_calls': active,
            'count': len(active)
        }), 200

    except Exception as e:
        logger.error(f"Error getting active calls: {e}")
        return jsonify({'error': str(e)}), 500