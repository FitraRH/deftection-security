<?php

namespace App\Services;

use App\Models\Scan;
use App\Models\User;
use App\Models\ScanDefect;
use Illuminate\Support\Str;
use Illuminate\Http\Request;
use Illuminate\Support\Carbon;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Log;
use Illuminate\Contracts\Pagination\LengthAwarePaginator;

class ScanHistoryService
{
    public function getScans(array $filters): LengthAwarePaginator
    {
        $canViewAll = auth()->user()->can('viewAny', Scan::class);

        $query = Scan::query()
            // Authorization-based user filtering - Use role as primary, policy as secondary
            ->when(!$canViewAll, function ($query) {
                // Regular users: Only see their own scans
                $query->where('scans.user_id', auth()->id());
                Log::info('Applied user restriction for user: ' . auth()->id());
            })
            // Admin user filter (when specific users are selected)
            ->when(!empty($filters['users']) && $canViewAll, function ($query) use ($filters) {
                $query->whereIn('scans.user_id', $filters['users']);
                Log::info('Applied admin user filter: ' . implode(',', $filters['users']));
            })
            // Admin role filter (when specific roles are selected)
            ->when(!empty($filters['roles']) && $canViewAll, function ($query) use ($filters) {
                $query->whereHas('user', function ($subQuery) use ($filters) {
                    $subQuery->whereIn('role', $filters['roles']);
                });
                Log::info('Applied admin role filter: ' . implode(',', $filters['roles']));
            })
            // Join users table only once and select specific columns
            ->join('users', 'scans.user_id', '=', 'users.id')
            // Use subquery for defect counts to avoid N+1 queries
            ->leftJoin(
                DB::raw('(SELECT scan_id, COUNT(*) as defect_count, GROUP_CONCAT(DISTINCT label ORDER BY label SEPARATOR ", ") as defect_labels FROM scan_defects GROUP BY scan_id) as defect_summary'),
                'scans.id',
                '=',
                'defect_summary.scan_id'
            )
            ->select([
                'scans.id',
                'scans.filename',
                'scans.original_path',
                'scans.annotated_path',
                'scans.is_defect',
                'scans.anomaly_score',
                'scans.anomaly_confidence_level',
                'scans.anomaly_inference_time_ms',
                'scans.classification_inference_time_ms',
                'scans.preprocessing_time_ms',
                'scans.postprocessing_time_ms',
                'scans.created_at',
                'scans.updated_at',
                'users.name as username',
                'users.role as user_role',
                // Pre-calculate formatted dates
                DB::raw('DATE_FORMAT(scans.created_at, "%d-%m-%Y") AS analysis_date'),
                DB::raw('DATE_FORMAT(scans.created_at, "%H:%i") AS analysis_time'),
                // Pre-calculate total processing time
                DB::raw('COALESCE(scans.anomaly_inference_time_ms, 0) + COALESCE(scans.classification_inference_time_ms, 0) + COALESCE(scans.preprocessing_time_ms, 0) + COALESCE(scans.postprocessing_time_ms, 0) AS total_processing_time'),
                // Get defect data from subquery
                DB::raw('COALESCE(defect_summary.defect_count, 0) AS total_defect'),
                DB::raw('COALESCE(defect_summary.defect_labels, "No defects") AS defect_scanned'),
                // Pre-calculate status
                DB::raw('CASE WHEN scans.is_defect = 1 THEN "defect" ELSE "good" END AS status')
            ])
            // Apply filters efficiently
            ->when(!empty($filters['status']), function ($query) use ($filters) {
                $statusValues = collect($filters['status'])->map(fn($s) => $s === 'defect' ? 1 : 0)->toArray();
                $query->whereIn('scans.is_defect', $statusValues);
            })
            ->when(!empty($filters['defectTypes']), function ($query) use ($filters) {
                $query->whereExists(function ($subquery) use ($filters) {
                    $subquery->select(DB::raw(1))
                        ->from('scan_defects')
                        ->whereColumn('scan_defects.scan_id', 'scans.id')
                        ->whereIn('scan_defects.label', $filters['defectTypes']);
                });
            })
            ->when(!empty($filters['search']), function ($query) use ($filters) {
                $searchTerm = trim($filters['search']);
                $query->where(function ($subQuery) use ($searchTerm) {
                    $subQuery->where('scans.filename', 'like', '%' . $searchTerm . '%')
                        ->orWhere('scans.anomaly_confidence_level', 'like', '%' . $searchTerm . '%')
                        ->orWhere('scans.anomaly_score', 'like', '%' . $searchTerm . '%')
                        ->orWhere('users.name', 'like', '%' . $searchTerm . '%')
                        ->orWhereExists(function ($subquery) use ($searchTerm) {
                            $subquery->select(DB::raw(1))
                                ->from('scan_defects')
                                ->whereColumn('scan_defects.scan_id', 'scans.id')
                                ->where('scan_defects.label', 'like', '%' . $searchTerm . '%');
                        });
                });
            })
            ->when(!empty($filters['dateFrom']), function ($query) use ($filters) {
                $query->whereDate('scans.created_at', '>=', $filters['dateFrom']);
            })
            ->when(!empty($filters['dateTo']), function ($query) use ($filters) {
                $query->whereDate('scans.created_at', '<=', $filters['dateTo']);
            });

        // Handle sorting with proper table prefixes
        $sortColumn = $filters['sortBy'];
        if (!str_contains($sortColumn, '.')) {
            // Special handling for user sorting
            if ($sortColumn === 'user') {
                $sortColumn = 'users.name';
            } else {
                $sortColumn = 'scans.' . $sortColumn;
            }
        }
        $query->orderBy($sortColumn, $filters['sortDir']);

        $result = $query->paginate($filters['perPage'], ['*'], 'page', $filters['page'])
            ->withQueryString()
            ->through(fn($scan) => [
                'id' => $scan->id,
                'filename' => $scan->filename,
                'original_path' => $scan->original_path,
                'annotated_path' => $scan->annotated_path,
                'status' => $scan->status,
                'anomaly_confidence_level' => $scan->anomaly_confidence_level,
                'anomaly_score' => number_format($scan->anomaly_score, 5),
                'analysis_date' => $scan->analysis_date,
                'analysis_time' => $scan->analysis_time,
                'total_processing_time' => number_format($scan->total_processing_time, 3),
                'defect_scanned' => $scan->defect_scanned,
                'total_defect' => $scan->total_defect,
                'username' => $scan->username,
                'user_role' => $scan->user_role,
                'created_at' => $scan->created_at->toISOString(),
                'updated_at' => $scan->updated_at->toISOString(),
            ]);

        Log::info('Pagination result', [
            'total' => $result->total(),
            'items_count' => count($result->items())
        ]);

        return $result;
    }

    /**
     * Get stable data checksum - SIMPLIFIED (defects never change independently)
     */
    public function getStableDataChecksum($userId): string
    {
        $canViewAll = auth()->user()->can('viewAny', Scan::class);

        // Only check scan data since defects are immutable and cascade with scans
        $scanStats = Scan::query()
            ->when(!$canViewAll, function ($query) use ($userId) {
                $query->where('user_id', $userId);
            })
            ->selectRaw('
                COUNT(*) as total_scans,
                MAX(updated_at) as latest_update
            ')
            ->first();

        $checksumData = [
            'user_role' => auth()->user()->role,
            'total_scans' => $scanStats->total_scans ?? 0,
            'latest_update' => $scanStats->latest_update
                ? Carbon::parse($scanStats->latest_update)->timestamp
                : 0,
        ];

        return md5(json_encode($checksumData));
    }

    /**
     * Get all defect types with counts (scoped by user access level) - OPTIMIZED
     */
    public function getDefectTypesWithCounts(): array
    {
        $canViewAll =  auth()->user()->can('viewAny', Scan::class);

        $query = DB::table('scan_defects')
            ->join('scans', 'scan_defects.scan_id', '=', 'scans.id')
            ->when(!$canViewAll, function ($query) {
                $query->where('scans.user_id', auth()->id());
            })
            ->select('scan_defects.label')
            ->selectRaw('COUNT(DISTINCT scan_defects.scan_id) as user_scan_count')
            ->groupBy('scan_defects.label')
            ->orderBy('user_scan_count', 'desc')
            ->orderBy('scan_defects.label', 'asc')
            ->get();

        return $query->map(function ($defect) {
            return [
                'label' => ucwords(str_replace(['_', '-'], ' ', $defect->label)),
                'value' => $defect->label,
                'count' => $defect->user_scan_count,
            ];
        })->toArray();
    }

    /**
     * Get status counts (scoped by user access level) - OPTIMIZED
     */
    public function getStatusCounts(): array
    {
        $isAdmin = auth()->user()->role === 'admin';
        $canViewAll = $isAdmin || auth()->user()->can('viewAny', Scan::class);

        $counts = Scan::query()
            ->when(!$canViewAll, function ($query) {
                $query->where('user_id', auth()->id());
            })
            ->selectRaw('
                SUM(CASE WHEN is_defect = 1 THEN 1 ELSE 0 END) as defect_count,
                SUM(CASE WHEN is_defect = 0 THEN 1 ELSE 0 END) as good_count
            ')
            ->first();

        return [
            [
                'label' => 'Good',
                'value' => 'good',
                'count' => $counts->good_count ?? 0,
            ],
            [
                'label' => 'Defect',
                'value' => 'defect',
                'count' => $counts->defect_count ?? 0,
            ]
        ];
    }

    /**
     * Get users with scan counts (admin only) - OPTIMIZED
     */
    public function getUsersWithCounts(): array
    {
        $canViewAll = auth()->user()->can('viewAny', Scan::class);

        // Only admins can access this
        if (!$canViewAll) {
            return [];
        }

        return DB::table('users')
            ->join('scans', 'users.id', '=', 'scans.user_id')
            ->select('users.id', 'users.name', 'users.email')
            ->selectRaw('COUNT(scans.id) as scans_count')
            ->groupBy('users.id', 'users.name', 'users.email')
            ->having('scans_count', '>', 0)
            ->orderBy('users.name')
            ->get()
            ->map(function ($user) {
                return [
                    'label' => $user->name, // Fixed: Just use name, not email
                    'value' => (string) $user->id,
                    'count' => (string) $user->scans_count,
                ];
            })
            ->toArray();
    }

    /**
     * Get roles with scan counts (admin only)
     */
    public function getRolesWithCounts(): array
    {
        $canViewAll = auth()->user()->can('viewAny', Scan::class);

        // Only admins can access this
        if (!$canViewAll) {
            return [];
        }

        return DB::table('users')
            ->join('scans', 'users.id', '=', 'scans.user_id')
            ->select('users.role')
            ->selectRaw('COUNT(scans.id) as scans_count')
            ->selectRaw('COUNT(DISTINCT users.id) as users_count')
            ->whereNotNull('users.role') // Exclude null roles
            ->where('users.role', '!=', '') // Exclude empty roles
            ->groupBy('users.role')
            ->having('scans_count', '>', 0) // Only roles with actual scans
            ->orderBy('users.role')
            ->get()
            ->map(function ($role) {
                return [
                    'label' => ucfirst($role->role) . ' (' . $role->users_count . ' users)',
                    'value' => $role->role,
                    'count' => $role->scans_count,
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
            'search' => 'nullable|string|max:255',
            'sort_by' => 'nullable|string|in:' . implode(',', array_merge(Scan::$sortable, ['user'])), // Add 'user' for sorting
            'sort_dir' => 'nullable|string|in:asc,desc',
            'per_page' => 'nullable|integer|min:1|max:100',
            'page' => 'nullable|integer|min:1',
            'status' => 'nullable|array',
            'status.*' => 'string|in:good,defect',
            'defect_types' => 'nullable|array',
            'defect_types.*' => 'string',
            'users' => 'nullable|array',
            'users.*' => 'integer|exists:users,id',
            'roles' => 'nullable|array', // Fixed: was 'role', should be 'roles'
            'roles.*' => 'string|in:admin,user', // Fixed: should match 'roles'
            'date_from' => 'nullable|date',
            'date_to' => 'nullable|date|after_or_equal:date_from',
        ]);

        $canFilterByUser = auth()->user()->can('filterByUser', Scan::class);

        // Only allow user and role filtering for admins
        $users = [];
        $roles = [];
        if ($canFilterByUser) {
            $users = $validated['users'] ?? [];
            $roles = $validated['roles'] ?? []; // Fixed: was using wrong key
        }

        return [
            'search' => $validated['search'] ?? '',
            'sortBy' => $validated['sort_by'] ?? 'created_at',
            'sortDir' => $validated['sort_dir'] ?? 'desc',
            'perPage' => $validated['per_page'] ?? 10,
            'page' => $validated['page'] ?? 1,
            'status' => $validated['status'] ?? [],
            'defectTypes' => $validated['defect_types'] ?? [],
            'users' => $users,
            'roles' => $roles, // Fixed: was using 'role', should be 'roles'
            'dateFrom' => $validated['date_from'] ?? null,
            'dateTo' => $validated['date_to'] ?? null,
        ];
    }
}
