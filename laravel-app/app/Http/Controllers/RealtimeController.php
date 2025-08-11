<?php

namespace App\Http\Controllers;

use Inertia\Inertia;
use Illuminate\Http\Request;
use App\Models\RealtimeSession;
use Illuminate\Support\Facades\Log;
use App\Services\RealtimeSessionQueryService;

class RealtimeController extends Controller
{
    public function __construct(
        protected RealtimeSessionQueryService    $realtimeSessionQueryService
    ) {}

    // list all realtime sessions
    public function index(Request $request)
    {
        try {
            // Get validated filters
            $filters = $this->realtimeSessionQueryService->filters($request);

            // Get paginated sessions data
            $sessions = $this->realtimeSessionQueryService->getSessions($filters);

            // Get filter options with counts for current user
            $sessionStatuses = $this->realtimeSessionQueryService->getSessionStatusesWithCounts();

            // Get user and role options only for authorized users
            $userOptions = auth()->user()->can('filterByUser', RealtimeSession::class)
                ? $this->realtimeSessionQueryService->getUsersWithCounts()
                : [];
            $rolesOptions = auth()->user()->can('filterByUser', RealtimeSession::class)
                ? $this->realtimeSessionQueryService->getRolesWithCounts()
                : [];


            // Generate initial checksum using the STABLE method
            $initialChecksum = $this->realtimeSessionQueryService->getStableDataChecksum(auth()->id());

            return Inertia::render('SessionHistory/Index', [
                'sessions' => $sessions->items(),
                'filters' => [
                    'current' => $filters,
                    'options' => [
                        'sessionStatuses' => $sessionStatuses,
                        'users' => $userOptions,
                        'roles' => $rolesOptions,
                        'perPageOptions' => [5, 10, 25, 50, 100],
                    ]
                ],
                'userCan' => [
                    'viewAllSessions' => auth()->user()->can('viewAny', RealtimeSession::class),
                    'filterByUser' => auth()->user()->can('filterByUser', RealtimeSession::class),
                ],
                'meta' => [
                    'total' => $sessions->total(),
                    'per_page' => $sessions->perPage(),
                    'current_page' => $sessions->currentPage(),
                    'last_page' => $sessions->lastPage(),
                    'from' => $sessions->firstItem(),
                    'to' => $sessions->lastItem(),
                ],
                'initialChecksum' => $initialChecksum,
            ]);
        } catch (\Exception $e) {
            // Log the error
            Log::error('Error in realtime session history index: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
                'filters' => $request->all(),
                'trace' => $e->getTraceAsString()
            ]);

            // Return error response
            return Inertia::render('SessionHistory/Index', [
                'sessions' => collect([]),
                'filters' => [
                    'current' => $this->realtimeSessionQueryService->filters($request),
                    'options' => [
                        'sessionStatuses' => [],
                        'users' => [],
                        'roles' => [],
                        'perPageOptions' => [5, 10, 25, 50, 100],
                    ]
                ],
                'userCan' => [
                    'viewAllSessions' => auth()->user()->can('viewAny', RealtimeSession::class),
                    'filterByUser' => auth()->user()->can('filterByUser', RealtimeSession::class),
                ],
                'error' => 'Failed to load session history. Please try again.',
                'meta' => [
                    'total' => 0,
                    'per_page' => 10,
                    'current_page' => 1,
                    'last_page' => 1,
                    'from' => null,
                    'to' => null,
                ],
                'initialChecksum' => '',
            ]);
        }
    }

    public function indexCheck(Request $request)
    {
        $startTime = microtime(true);

        try {
            $lastChecksum = $request->input('checksum', '');
            $userId = auth()->id();
            $currentChecksum = $this->realtimeSessionQueryService->getStableDataChecksum($userId);
            $hasChanges = $lastChecksum !== $currentChecksum;

            return response()->json([
                'has_changes' => $hasChanges,
                'checksum' => $currentChecksum,
                'timestamp' => now()->toISOString(),
                'response_time_ms' => round((microtime(true) - $startTime) * 1000, 2),
            ]);
        } catch (\Exception $e) {
            Log::error('Error checking session updates: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
                'last_checksum' => $request->input('checksum', ''),
            ]);

            return response()->json([
                'error' => 'Failed to check updates',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    public function indexApi(Request $request)
    {
        $startTime = microtime(true);

        try {
            // Get validated filters
            $filters = $this->realtimeSessionQueryService->filters($request);

            // Get paginated sessions data
            $sessions = $this->realtimeSessionQueryService->getSessions($filters);

            // Get filter options with counts for current user
            $sessionStatuses = $this->realtimeSessionQueryService->getSessionStatusesWithCounts();

            $userOptions = $this->realtimeSessionQueryService->getUsersWithCounts();

            $rolesOptions = $this->realtimeSessionQueryService->getRolesWithCounts();

            // Get current STABLE checksum
            $currentChecksum = $this->realtimeSessionQueryService->getStableDataChecksum(auth()->id());

            $data = [
                'sessions' => $sessions->items(),
                'filters' => [
                    'current' => $filters,
                    'options' => [
                        'sessionStatuses' => $sessionStatuses,
                        'users' => $userOptions,
                        'roles' => $rolesOptions,
                        'perPageOptions' => [5, 10, 25, 50, 100],
                    ]
                ],
                'userCan' => [
                    'viewAllSessions' => auth()->user()->can('viewAny', RealtimeSession::class),
                    'filterByUser' => auth()->user()->can('filterByUser', RealtimeSession::class),
                ],
                'meta' => [
                    'total' => $sessions->total(),
                    'per_page' => $sessions->perPage(),
                    'current_page' => $sessions->currentPage(),
                    'last_page' => $sessions->lastPage(),
                    'from' => $sessions->firstItem(),
                    'to' => $sessions->lastItem(),
                ],
                'checksum' => $currentChecksum,
                'timestamp' => now()->toISOString(),
                'response_time_ms' => round((microtime(true) - $startTime) * 1000, 2)
            ];

            return response()->json($data);
        } catch (\Exception $e) {
            Log::error('Error in session poll data: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
                'filters' => $request->all(),
            ]);

            return response()->json([
                'error' => 'Failed to fetch session data',
                'timestamp' => now()->toISOString()
            ], 500);
        }
    }

    public function indexRefresh(Request $request)
    {
        try {
            // This is essentially the same as indexApi but ensures fresh data
            return $this->indexApi($request);
        } catch (\Exception $e) {
            Log::error('Error in session force refresh: ' . $e->getMessage(), [
                'user_id' => auth()->id(),
                'filters' => $request->all(),
            ]);

            return response()->json([
                'error' => 'Failed to refresh session data. Please try again.',
                'meta' => [
                    'total' => 0,
                    'per_page' => 10,
                    'current_page' => 1,
                    'last_page' => 1,
                    'from' => null,
                    'to' => null,
                ],
                'initialChecksum' => '',
            ]);
        }
    }

    // page for realtime detection
    public function create()
    {
        return Inertia::render('RealTimeAnalysis/Index', []);
    }

    // delete a specific realtime session
    public function destroy(RealtimeSession $realtimeSession)
    {
        try {
            // Authorize the deletion
            $this->authorize('delete', $realtimeSession);

            // Delete the session (this will cascade to related RealtimeScans and RealtimeScanDefects)
            $realtimeSession->delete();

            return redirect()->back()->with([
                'message' => 'Session deleted successfully',
            ]);
        } catch (\Exception $e) {
            Log::error('Error deleting realtime session: ' . $e->getMessage(), [
                'session_id' => $realtimeSession->id,
                'user_id' => auth()->id(),
            ]);

            return redirect()->back()->with([
                'error' => 'Failed to delete session',
            ]);
        }
    }
}
