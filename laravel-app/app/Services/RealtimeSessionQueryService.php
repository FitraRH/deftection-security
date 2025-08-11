<?php

namespace App\Services;

use App\Models\User;
use App\Models\RealtimeSession;
use App\Models\RealtimeScanDefect;
use Illuminate\Support\Str;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class RealtimeSessionQueryService
{
    public function getSessions(array $filters): LengthAwarePaginator
    {
        $canViewAll = auth()->user()->can('viewAny', RealtimeSession::class);

        $query = RealtimeSession::query()
            // Join with users table for user information
            ->leftJoin('users', 'realtime_sessions.user_id', '=', 'users.id')
            ->select([
                'realtime_sessions.id',
                'realtime_sessions.session_status',
                'realtime_sessions.session_start',
                'realtime_sessions.duration_seconds',
                'realtime_sessions.total_frames_processed',
                'realtime_sessions.defect_rate',
                'realtime_sessions.created_at',
                'realtime_sessions.updated_at',
                'users.name as username',
                'users.role as user_role',
                DB::raw('DATE(realtime_sessions.session_start) as session_date'),
                DB::raw('CASE 
                    WHEN realtime_sessions.session_status = "completed" THEN "completed"
                    WHEN realtime_sessions.session_status = "active" THEN "active"
                    WHEN realtime_sessions.session_status = "paused" THEN "paused"
                    WHEN realtime_sessions.session_status = "aborted" THEN "aborted"
                    ELSE "unknown"
                END as status')
            ])
            // Authorization-based user filtering
            ->when(!$canViewAll, function ($query) {
                // Regular users: Only see their own sessions
                $query->where('realtime_sessions.user_id', auth()->id());
                Log::info('Applied user restriction for user: ' . auth()->id());
            })
            // Admin user filter (when specific users are selected)
            ->when(!empty($filters['users']) && $canViewAll, function ($query) use ($filters) {
                $query->whereIn('realtime_sessions.user_id', $filters['users']);
                Log::info('Applied admin user filter: ' . implode(',', $filters['users']));
            })
            // Admin role filter (when specific roles are selected)
            ->when(!empty($filters['roles']) && $canViewAll, function ($query) use ($filters) {
                $query->whereHas('user', function ($subQuery) use ($filters) {
                    $subQuery->whereIn('role', $filters['roles']);
                });
                Log::info('Applied admin role filter: ' . implode(',', $filters['roles']));
            })
            // Session status filter
            ->when(!empty($filters['sessionStatuses']), function ($query) use ($filters) {
                $query->whereIn('realtime_sessions.session_status', $filters['sessionStatuses']);
                Log::info('Applied session status filter: ' . implode(',', $filters['sessionStatuses']));
            })
            // Date range filter
            ->when(!empty($filters['dateFrom']), function ($query) use ($filters) {
                $query->where('realtime_sessions.session_start', '>=', $filters['dateFrom'] . ' 00:00:00');
                Log::info('Applied date from filter: ' . $filters['dateFrom']);
            })
            ->when(!empty($filters['dateTo']), function ($query) use ($filters) {
                $query->where('realtime_sessions.session_start', '<=', $filters['dateTo'] . ' 23:59:59');
                Log::info('Applied date to filter: ' . $filters['dateTo']);
            });

        // Sorting
        $sortColumn = $filters['sortBy'];
        if (in_array($sortColumn, ['id', 'session_start', 'duration_seconds', 'total_frames_processed', 'defect_rate', 'session_status', 'created_at', 'updated_at'])) {
            $sortColumn = 'realtime_sessions.' . $sortColumn;
        } elseif ($sortColumn === 'user') {
            $sortColumn = 'users.name';
        }
        $query->orderBy($sortColumn, $filters['sortDir']);

        $result = $query->paginate($filters['perPage'], ['*'], 'page', $filters['page'])
            ->withQueryString()
            ->through(fn($session) => [
                'id' => $session->id,
                'session_date' => $session->session_date,
                'duration_seconds' => $session->duration_seconds,
                'total_scans' => $session->total_frames_processed,
                'defect_rate' => number_format($session->defect_rate ?? 0, 2) . '%',
                'status' => ucfirst($session->status),
                'session_start' => $session->session_start,
                'username' => $session->username,
                'user_role' => $session->user_role,
                'created_at' => $session->created_at->toISOString(),
                'updated_at' => $session->updated_at->toISOString(),
            ]);

        Log::info('Session pagination result', [
            'total' => $result->total(),
            'items_count' => count($result->items())
        ]);

        return $result;
    }

    /**
     * Get stable data checksum - SIMPLIFIED (sessions and related data)
     */
    public function getStableDataChecksum($userId): string
    {
        $canViewAll = auth()->user()->can('viewAny', RealtimeSession::class);

        // Check session data since realtime scans are immutable and cascade with sessions
        $sessionStats = RealtimeSession::query()
            ->when(!$canViewAll, function ($query) use ($userId) {
                $query->where('user_id', $userId);
            })
            ->selectRaw('
                COUNT(*) as total_sessions,
                MAX(updated_at) as latest_update
            ')
            ->first();

        $checksumData = [
            'user_role' => auth()->user()->role,
            'total_sessions' => $sessionStats->total_sessions ?? 0,
            'latest_update' => $sessionStats->latest_update
                ? Carbon::parse($sessionStats->latest_update)->timestamp
                : 0,
        ];

        return md5(json_encode($checksumData));
    }

    /**
     * Get all session statuses with counts (scoped by user access level)
     */
    public function getSessionStatusesWithCounts(): array
    {
        $canViewAll = auth()->user()->can('viewAny', RealtimeSession::class);

        $counts = RealtimeSession::query()
            ->when(!$canViewAll, function ($query) {
                $query->where('user_id', auth()->id());
            })
            ->selectRaw('
                session_status,
                COUNT(*) as session_count
            ')
            ->whereNotNull('session_status')
            ->groupBy('session_status')
            ->orderBy('session_status')
            ->get();

        return $counts->map(function ($status) {
            return [
                'label' => ucfirst($status->session_status),
                'value' => $status->session_status,
                'count' => $status->session_count,
            ];
        })->toArray();
    }

    /**
     * Get users with session counts (admin only)
     */
    public function getUsersWithCounts(): array
    {
        $canViewAll = auth()->user()->can('viewAny', RealtimeSession::class);

        // Only admins can access this
        if (!$canViewAll) {
            return [];
        }

        return DB::table('users')
            ->join('realtime_sessions', 'users.id', '=', 'realtime_sessions.user_id')
            ->select('users.id', 'users.name', 'users.email')
            ->selectRaw('COUNT(realtime_sessions.id) as sessions_count')
            ->groupBy('users.id', 'users.name', 'users.email')
            ->having('sessions_count', '>', 0)
            ->orderBy('users.name')
            ->get()
            ->map(function ($user) {
                return [
                    'label' => $user->name,
                    'value' => (string) $user->id,
                    'count' => (string) $user->sessions_count,
                ];
            })
            ->toArray();
    }

    /**
     * Get roles with session counts (admin only)
     */
    public function getRolesWithCounts(): array
    {
        $canViewAll = auth()->user()->can('viewAny', RealtimeSession::class);

        // Only admins can access this
        if (!$canViewAll) {
            return [];
        }

        return DB::table('users')
            ->join('realtime_sessions', 'users.id', '=', 'realtime_sessions.user_id')
            ->select('users.role')
            ->selectRaw('COUNT(realtime_sessions.id) as sessions_count')
            ->selectRaw('COUNT(DISTINCT users.id) as users_count')
            ->whereNotNull('users.role')
            ->where('users.role', '!=', '')
            ->groupBy('users.role')
            ->having('sessions_count', '>', 0)
            ->orderBy('users.role')
            ->get()
            ->map(function ($role) {
                return [
                    'label' => ucfirst($role->role) . ' (' . $role->users_count . ' users)',
                    'value' => $role->role,
                    'count' => $role->sessions_count,
                ];
            })
            ->toArray();
    }

    /**
     * Get and validate filters from request
     */
    public function filters(Request $request): array
    {

        // Validate request parameters
        $validated = $request->validate([
            'sort_by' => 'nullable|string|in:' . implode(',', RealtimeSession::$sortable),
            'sort_dir' => 'nullable|string|in:asc,desc',
            'per_page' => 'nullable|integer|min:1|max:100',
            'page' => 'nullable|integer|min:1',
            'session_statuses' => 'nullable|array',
            'session_statuses.*' => 'string|in:completed,aborted,active,paused',
            'users' => 'nullable|array',
            'users.*' => 'integer|exists:users,id',
            'roles' => 'nullable|array',
            'roles.*' => 'string|in:admin,user',
            'date_from' => 'nullable|date',
            'date_to' => 'nullable|date|after_or_equal:date_from',
        ]);

        $canFilterByUser = auth()->user()->can('filterByUser', RealtimeSession::class);

        // Only allow user and role filtering for admins
        $users = [];
        $roles = [];
        if ($canFilterByUser) {
            $users = $validated['users'] ?? [];
            $roles = $validated['roles'] ?? [];
        }

        return [
            'sortBy' => $validated['sort_by'] ?? 'session_start',
            'sortDir' => $validated['sort_dir'] ?? 'desc',
            'perPage' => $validated['per_page'] ?? 10,
            'page' => $validated['page'] ?? 1,
            'sessionStatuses' => $validated['session_statuses'] ?? [],
            'users' => $users,
            'roles' => $roles,
            'dateFrom' => $validated['date_from'] ?? null,
            'dateTo' => $validated['date_to'] ?? null,
        ];
    }
}
