<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\RealtimeSession;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Http;
use Illuminate\Validation\ValidationException;
use Illuminate\Auth\Access\AuthorizationException;
use Illuminate\Database\Eloquent\ModelNotFoundException;

class RealtimeAnalysisController extends Controller
{
    //
    public function startSession(Request $request)
    {
        try {
            // Validate request
            $validated = $request->validate([
                'camera_location' => 'nullable|string|max:255',
                'session_config' => 'nullable|array'
            ]);

            // Check Flask API health first
            $flaskUrl = env('FLASK_API_URL', 'http://localhost:5000');

            try {
                $healthResponse = Http::timeout(10)->get($flaskUrl . '/api/health');

                if (!$healthResponse->successful()) {
                    return response()->json([
                        'error' => 'Flask API server is not available',
                        'message' => 'Unable to start session. The detection service is currently unavailable.',
                        'status' => 'failed',
                        'timestamp' => now()->toISOString()
                    ], 503);
                }

                $healthData = $healthResponse->json();
                if ($healthData['status'] !== 'healthy' && $healthData['status'] !== 'ok') {
                    return response()->json([
                        'error' => 'Flask API server is unhealthy',
                        'message' => 'Unable to start session. The detection service is not functioning properly.',
                        'status' => 'failed',
                        'timestamp' => now()->toISOString()
                    ], 503);
                }
            } catch (\Exception $e) {
                Log::error('Flask API health check failed: ' . $e->getMessage(), [
                    'flask_url' => $flaskUrl,
                    'user_id' => auth()->id(),
                ]);

                return response()->json([
                    'error' => 'Flask API connection failed',
                    'message' => 'Unable to connect to the detection service. Please try again later.',
                    'status' => 'failed',
                    'timestamp' => now()->toISOString()
                ], 503);
            }

            // Check if user already has an active session
            $existingActiveSession = RealtimeSession::where('user_id', auth()->id())
                ->where('session_status', 'active')
                ->first();

            if ($existingActiveSession) {
                return response()->json([
                    'error' => 'Active session exists',
                    'message' => 'You already have an active session. Please stop the current session before starting a new one.',
                    'status' => 'failed',
                    'existing_session_id' => $existingActiveSession->id,
                    'timestamp' => now()->toISOString()
                ], 409);
            }

            // Authorize session creation
            $this->authorize('create', RealtimeSession::class);

            // Create new session
            $session = RealtimeSession::create([
                'user_id' => auth()->id(),
                'session_status' => 'active',
                'session_start' => now(),
                'camera_location' => $validated['camera_location'] ?? null,
                'total_frames_processed' => 0,
                'defect_count' => 0,
                'good_count' => 0,
                'defect_rate' => 0.00,
                'good_rate' => 0.00,
                'throughput_fps' => 0.000,
            ]);

            Log::info('Realtime session started successfully', [
                'session_id' => $session->id,
                'user_id' => auth()->id(),
                'camera_location' => $validated['camera_location'] ?? 'Not specified',
                'flask_api_health' => $healthData
            ]);

            return response()->json([
                'success' => true,
                'message' => 'Session started successfully',
                'session' => [
                    'id' => $session->id,
                    'session_status' => $session->session_status,
                    'session_start' => $session->session_start->toISOString(),
                    'camera_location' => $session->camera_location,
                    'user_id' => $session->user_id
                ],
                'flask_api' => [
                    'status' => 'healthy',
                    'url' => $flaskUrl
                ],
                'timestamp' => now()->toISOString()
            ], 201);
        } catch (ValidationException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Invalid input data provided',
                'errors' => $e->errors(),
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 422);
        } catch (AuthorizationException $e) {
            return response()->json([
                'error' => 'Unauthorized',
                'message' => 'You are not authorized to create sessions',
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 403);
        } catch (\Exception $e) {
            Log::error('Error starting realtime session: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
                'request_data' => $request->all(),
                'trace' => $e->getTraceAsString()
            ]);

            return response()->json([
                'error' => 'Session start failed',
                'message' => 'Unable to start session due to an internal error. Please try again.',
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }
    /**
     * Pause an active realtime session
     */
    public function pauseSession(Request $request, $sessionId = null)
    {
        try {
            // If no session ID provided, try to find user's active session
            if (!$sessionId) {
                $session = RealtimeSession::where('user_id', auth()->id())
                    ->where('session_status', 'active')
                    ->first();

                if (!$session) {
                    return response()->json([
                        'error' => 'No active session found',
                        'message' => 'You do not have any active sessions to pause.',
                        'status' => 'failed',
                        'timestamp' => now()->toISOString()
                    ], 404);
                }
            } else {
                $session = RealtimeSession::findOrFail($sessionId);
            }

            // Authorize session update
            $this->authorize('update', $session);

            // Check if session can be paused
            if ($session->session_status !== 'active') {
                return response()->json([
                    'error' => 'Session cannot be paused',
                    'message' => 'Only active sessions can be paused.',
                    'status' => 'failed',
                    'current_status' => $session->session_status,
                    'timestamp' => now()->toISOString()
                ], 409);
            }

            // Update session to paused
            $session->update(['session_status' => 'paused']);

            Log::info('Realtime session paused successfully', [
                'session_id' => $session->id,
                'user_id' => auth()->id(),
            ]);

            return response()->json([
                'success' => true,
                'message' => 'Session paused successfully',
                'session' => [
                    'id' => $session->id,
                    'session_status' => $session->session_status,
                    'session_start' => $session->session_start->toISOString(),
                    'total_frames_processed' => $session->total_frames_processed,
                    'defect_count' => $session->defect_count,
                    'good_count' => $session->good_count,
                    'defect_rate' => $session->defect_rate,
                ],
                'timestamp' => now()->toISOString()
            ], 200);
        } catch (\Exception $e) {
            Log::error('Error pausing realtime session: ' . $e->getMessage(), [
                'session_id' => $sessionId,
                'user_id' => auth()->id(),
            ]);

            return response()->json([
                'error' => 'Session pause failed',
                'message' => 'Unable to pause session due to an internal error.',
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    /**
     * Resume a paused realtime session
     */
    public function resumeSession(Request $request, $sessionId = null)
    {
        try {
            // If no session ID provided, try to find user's paused session
            if (!$sessionId) {
                $session = RealtimeSession::where('user_id', auth()->id())
                    ->where('session_status', 'paused')
                    ->first();

                if (!$session) {
                    return response()->json([
                        'error' => 'No paused session found',
                        'message' => 'You do not have any paused sessions to resume.',
                        'status' => 'failed',
                        'timestamp' => now()->toISOString()
                    ], 404);
                }
            } else {
                $session = RealtimeSession::findOrFail($sessionId);
            }

            // Authorize session update
            $this->authorize('update', $session);

            // Check if session can be resumed
            if ($session->session_status !== 'paused') {
                return response()->json([
                    'error' => 'Session cannot be resumed',
                    'message' => 'Only paused sessions can be resumed.',
                    'status' => 'failed',
                    'current_status' => $session->session_status,
                    'timestamp' => now()->toISOString()
                ], 409);
            }

            // Update session to active
            $session->update(['session_status' => 'active']);

            Log::info('Realtime session resumed successfully', [
                'session_id' => $session->id,
                'user_id' => auth()->id(),
            ]);

            return response()->json([
                'success' => true,
                'message' => 'Session resumed successfully',
                'session' => [
                    'id' => $session->id,
                    'session_status' => $session->session_status,
                    'session_start' => $session->session_start->toISOString(),
                    'total_frames_processed' => $session->total_frames_processed,
                    'defect_count' => $session->defect_count,
                    'good_count' => $session->good_count,
                    'defect_rate' => $session->defect_rate,
                ],
                'timestamp' => now()->toISOString()
            ], 200);
        } catch (\Exception $e) {
            Log::error('Error resuming realtime session: ' . $e->getMessage(), [
                'session_id' => $sessionId,
                'user_id' => auth()->id(),
            ]);

            return response()->json([
                'error' => 'Session resume failed',
                'message' => 'Unable to resume session due to an internal error.',
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    /**
     * Stop an active realtime session
     */
    public function stopSession(Request $request, $sessionId = null)
    {
        try {
            // If no session ID provided, try to find user's active session
            if (!$sessionId) {
                $session = RealtimeSession::where('user_id', auth()->id())
                    ->where('session_status', 'active')
                    ->first();

                if (!$session) {
                    return response()->json([
                        'error' => 'No active session found',
                        'message' => 'You do not have any active sessions to stop.',
                        'status' => 'failed',
                        'timestamp' => now()->toISOString()
                    ], 404);
                }
            } else {
                $session = RealtimeSession::findOrFail($sessionId);
            }

            // Authorize session update
            $this->authorize('update', $session);

            // Check if session can be stopped (active or paused)
            if (!in_array($session->session_status, ['active', 'paused'])) {
                return response()->json([
                    'error' => 'Session cannot be stopped',
                    'message' => 'Only active or paused sessions can be stopped.',
                    'status' => 'failed',
                    'current_status' => $session->session_status,
                    'timestamp' => now()->toISOString()
                ], 409);
            }

            // Calculate session duration
            $sessionEnd = now();
            $durationSeconds = $session->session_start->diffInSeconds($sessionEnd);

            // Update session with completion data
            $session->update([
                'session_status' => 'completed',
                'session_end' => $sessionEnd,
                'duration_seconds' => $durationSeconds,
                // Recalculate rates based on current counts
                'defect_rate' => $session->total_frames_processed > 0
                    ? round(($session->defect_count / $session->total_frames_processed) * 100, 2)
                    : 0.00,
                'good_rate' => $session->total_frames_processed > 0
                    ? round(($session->good_count / $session->total_frames_processed) * 100, 2)
                    : 0.00,
                'throughput_fps' => $durationSeconds > 0
                    ? round($session->total_frames_processed / $durationSeconds, 3)
                    : 0.000
            ]);

            Log::info('Realtime session stopped successfully', [
                'session_id' => $session->id,
                'user_id' => auth()->id(),
                'duration_seconds' => $durationSeconds,
                'total_frames' => $session->total_frames_processed,
                'defect_rate' => $session->defect_rate
            ]);

            return response()->json([
                'success' => true,
                'message' => 'Session stopped successfully',
                'session' => [
                    'id' => $session->id,
                    'session_status' => $session->session_status,
                    'session_start' => $session->session_start->toISOString(),
                    'session_end' => $session->session_end->toISOString(),
                    'duration_seconds' => $session->duration_seconds,
                    'total_frames_processed' => $session->total_frames_processed,
                    'defect_count' => $session->defect_count,
                    'good_count' => $session->good_count,
                    'defect_rate' => $session->defect_rate,
                    'good_rate' => $session->good_rate,
                    'throughput_fps' => $session->throughput_fps
                ],
                'timestamp' => now()->toISOString()
            ], 200);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Session not found',
                'message' => 'The specified session could not be found.',
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 404);
        } catch (AuthorizationException $e) {
            return response()->json([
                'error' => 'Unauthorized',
                'message' => 'You are not authorized to stop this session.',
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 403);
        } catch (\Exception $e) {
            Log::error('Error stopping realtime session: ' . $e->getMessage(), [
                'session_id' => $sessionId,
                'user_id' => auth()->id(),
                'trace' => $e->getTraceAsString()
            ]);

            return response()->json([
                'error' => 'Session stop failed',
                'message' => 'Unable to stop session due to an internal error. Please try again.',
                'status' => 'failed',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    /**
     * Get current active or paused session for user
     */
    public function getCurrentSession(Request $request)
    {
        try {
            $session = RealtimeSession::where('user_id', auth()->id())
                ->whereIn('session_status', ['active', 'paused'])
                ->first();

            if (!$session) {
                return response()->json([
                    'session' => null,
                    'message' => 'No active or paused session found',
                    'timestamp' => now()->toISOString()
                ], 200);
            }

            return response()->json([
                'session' => [
                    'id' => $session->id,
                    'session_status' => $session->session_status,
                    'session_start' => $session->session_start->toISOString(),
                    'camera_location' => $session->camera_location,
                    'total_frames_processed' => $session->total_frames_processed,
                    'defect_count' => $session->defect_count,
                    'good_count' => $session->good_count,
                    'defect_rate' => $session->defect_rate,
                    'duration_seconds' => now()->diffInSeconds($session->session_start)
                ],
                'timestamp' => now()->toISOString()
            ], 200);
        } catch (\Exception $e) {
            Log::error('Error getting current session: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
            ]);

            return response()->json([
                'error' => 'Failed to get current session',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }
}
