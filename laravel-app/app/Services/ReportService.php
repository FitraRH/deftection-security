<?php

namespace App\Services;

use App\Models\Scan;
use App\Models\DefectType;
use App\Models\User;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\DB;
use Carbon\Carbon;

class ReportService
{
    protected ChartGenerationService $chartService;

    public function __construct(ChartGenerationService $chartService)
    {
        $this->chartService = $chartService;
    }

    /**
     * Prepare data for single scan report
     */
    public function prepareSingleScanReport(Scan $scan): array
    {
        // Load relationships - they might be empty collections/null
        $scan->load(['user', 'scanDefects', 'scanThreat']);

        // Get defect types for explanations
        $defectTypes = DefectType::all()->keyBy('slug');

        // Process defects with explanations (handle empty collection)
        $processedDefects = $this->processDefectsWithExplanations($scan->scanDefects ?? collect(), $defectTypes);

        // Generate charts data
        $chartsData = $this->generateSingleScanCharts($scan);

        // Generate chart images using the chart service
        $chartImages = $this->generateChartImages($chartsData);

        // Get image paths
        $imagePaths = $this->getImagePaths($scan);

        return [
            'scan' => $scan,
            'defects' => $processedDefects,  // Will be empty collection if no defects
            'chartsData' => $chartsData,
            'chartImages' => $chartImages, // Base64 chart images
            'imagePaths' => $imagePaths,
            'summary' => $this->generateScanSummary($scan),
            'generatedAt' => now(),
            'reportTitle' => "Scan Analysis Report - {$scan->filename}",
        ];
    }

    /**
     * Prepare data for batch scan report
     */
    public function prepareBatchScanReport(array $filters): array
    {
        // Get scans based on filters with proper authorization
        $scans = $this->getBatchScansData($filters);

        // Log the scan filtering results
        Log::info('Batch report scan filtering results', [
            'total_scans_found' => $scans->count(),
            'filters_applied' => $filters,
            'user_can_view_all' => auth()->user()->can('viewAny', Scan::class)
        ]);

        // Generate comprehensive statistics
        $statistics = $this->generateBatchStatistics($scans, $filters);

        // Generate charts data for batch analysis
        $chartsData = $this->generateBatchChartsData($scans, $filters);

        // Generate chart images using the chart service
        $chartImages = $this->generateBatchChartImages($chartsData);

        // Get defect types for explanations
        $defectTypes = DefectType::all()->keyBy('slug');

        // Process detailed defect analysis
        $defectAnalysis = $this->generateBatchDefectAnalysis($scans, $defectTypes);

        // Generate user-specific analysis (if admin report)
        $userAnalysis = $this->generateUserAnalysis($scans, $filters);

        return [
            'scans' => $scans->take(300), // Limit detailed scans to prevent PDF bloat
            'statistics' => $statistics,
            'chartsData' => $chartsData,
            'chartImages' => $chartImages,
            'defectAnalysis' => $defectAnalysis,
            'userAnalysis' => $userAnalysis,
            'filters' => $filters,
            'summary' => $this->generateBatchSummary($scans, $filters),
            'generatedAt' => now(),
            'reportTitle' => $this->generateBatchReportTitle($filters),
        ];
    }

    /**
     * Generate chart images for single scan
     */
    private function generateChartImages(array $chartsData): array
    {
        $chartImages = [];

        try {
            // Defect distribution chart
            if (!empty($chartsData['defect_distribution'])) {
                $chartImages['defect_distribution'] = $this->chartService->generateDefectDistributionChart(
                    $chartsData['defect_distribution']
                );
            }

            // Severity distribution chart
            if (!empty($chartsData['severity_distribution'])) {
                $chartImages['severity_distribution'] = $this->chartService->generateSeverityDistributionChart(
                    $chartsData['severity_distribution']
                );
            }

            // Processing time chart
            if (!empty($chartsData['processing_time_breakdown'])) {
                $chartImages['processing_time'] = $this->chartService->generateProcessingTimeChart(
                    $chartsData['processing_time_breakdown']
                );
            }

            Log::info('Single scan chart images generated', [
                'charts_generated' => array_keys(array_filter($chartImages)),
                'total_charts' => count($chartImages)
            ]);
        } catch (\Exception $e) {
            Log::error('Failed to generate single scan chart images', [
                'error' => $e->getMessage(),
                'charts_data' => array_keys($chartsData)
            ]);
        }

        return $chartImages;
    }

    /**
     * Generate chart images for batch report
     */
    private function generateBatchChartImages(array $chartsData): array
    {
        $chartImages = [];

        try {
            // Daily trends chart
            if (!empty($chartsData['daily_scan_trends'])) {
                $chartImages['daily_trends'] = $this->chartService->generateDailyScanTrendsChart(
                    $chartsData['daily_scan_trends']
                );
            }

            // Anomaly scores chart
            if (!empty($chartsData['anomaly_score_distribution'])) {
                $chartImages['anomaly_scores'] = $this->chartService->generateAnomalyScoreChart(
                    $chartsData['anomaly_score_distribution']
                );
            }

            // User activity chart
            if (!empty($chartsData['user_activity_analysis'])) {
                $chartImages['user_activity'] = $this->chartService->generateUserActivityChart(
                    $chartsData['user_activity_analysis']
                );
            }

            // Confidence levels chart
            if (!empty($chartsData['confidence_level_distribution'])) {
                $chartImages['confidence_levels'] = $this->chartService->generateConfidenceLevelChart(
                    $chartsData['confidence_level_distribution']
                );
            }

            // Processing performance chart
            if (!empty($chartsData['processing_time_analysis'])) {
                $chartImages['processing_performance'] = $this->chartService->generateProcessingPerformanceChart(
                    $chartsData['processing_time_analysis']
                );
            }

            // Hourly patterns chart
            if (!empty($chartsData['hourly_scan_patterns'])) {
                $chartImages['hourly_patterns'] = $this->chartService->generateHourlyPatternsChart(
                    $chartsData['hourly_scan_patterns']
                );
            }

            // Defect distribution chart
            if (!empty($chartsData['defect_type_distribution'])) {
                $chartImages['defect_distribution'] = $this->chartService->generateDefectDistributionChart(
                    $chartsData['defect_type_distribution']
                );
            }

            // Severity distribution chart
            if (!empty($chartsData['severity_distribution'])) {
                $chartImages['severity_distribution'] = $this->chartService->generateSeverityDistributionChart(
                    $chartsData['severity_distribution']
                );
            }

            Log::info('Batch report chart images generated', [
                'charts_generated' => array_keys(array_filter($chartImages)),
                'total_charts' => count($chartImages)
            ]);
        } catch (\Exception $e) {
            Log::error('Failed to generate batch report chart images', [
                'error' => $e->getMessage(),
                'charts_data' => array_keys($chartsData)
            ]);
        }

        return $chartImages;
    }

    /**
     * Get scans data based on batch filters with proper authorization
     */
    private function getBatchScansData(array $filters): Collection
    {
        $query = Scan::with(['user', 'scanDefects', 'scanThreat'])
            ->whereBetween('created_at', [$filters['dateFrom'], $filters['dateTo']]);

        // Apply authorization-based user filtering
        if (!auth()->user()->can('viewAny', Scan::class)) {
            // Regular users: Only see their own scans
            $query->where('user_id', auth()->id());
            Log::info('Applied user restriction for regular user', [
                'user_id' => auth()->id()
            ]);
        } elseif (!empty($filters['users'])) {
            // Admin with specific user filter
            $query->whereIn('user_id', $filters['users']);
            Log::info('Applied admin user filter', [
                'filtered_users' => $filters['users']
            ]);
        }
        // If admin with no user filter, show all scans

        // Apply role filter (admin only)
        if (!empty($filters['roles']) && auth()->user()->can('viewAny', Scan::class)) {
            $query->whereHas('user', function ($q) use ($filters) {
                $q->whereIn('role', $filters['roles']);
            });
            Log::info('Applied role filter', [
                'filtered_roles' => $filters['roles']
            ]);
        }

        // Apply status filter
        if ($filters['status'] !== 'all') {
            $isDefect = $filters['status'] === 'defective';
            $query->where('is_defect', $isDefect);
            Log::info('Applied status filter', [
                'status' => $filters['status'],
                'is_defect' => $isDefect
            ]);
        }

        // Apply defect type filter
        if (!empty($filters['defectTypes'])) {
            $query->whereHas('scanDefects', function ($q) use ($filters) {
                $q->whereIn('label', $filters['defectTypes']);
            });
            Log::info('Applied defect type filter', [
                'defect_types' => $filters['defectTypes']
            ]);
        }

        $result = $query->orderBy('created_at', 'desc')->get();

        Log::info('Final scan query results', [
            'total_found' => $result->count(),
            'sample_scan_ids' => $result->take(5)->pluck('id')->toArray()
        ]);

        return $result;
    }

    /**
     * Generate batch report title based on filters
     */
    private function generateBatchReportTitle(array $filters): string
    {
        $title = "Batch Scan Analysis Report";

        if (!empty($filters['users']) && auth()->user()->can('viewAny', Scan::class)) {
            $userCount = count($filters['users']);
            if ($userCount === 1) {
                $user = User::find($filters['users'][0]);
                $title .= " - " . ($user ? $user->name : "User ID {$filters['users'][0]}");
            } else {
                $title .= " - {$userCount} Users";
            }
        } elseif (!auth()->user()->can('viewAny', Scan::class)) {
            $title .= " - " . auth()->user()->name;
        }

        return $title;
    }

    /**
     * Generate batch summary
     */
    private function generateBatchSummary(Collection $scans, array $filters): array
    {
        $canViewAll = auth()->user()->can('viewAny', Scan::class);

        $scope = "Personal Report";
        if ($canViewAll) {
            if (!empty($filters['users'])) {
                $userCount = count($filters['users']);
                $scope = $userCount === 1 ? "Single User Report" : "{$userCount} Users Report";
            } else {
                $scope = "System-wide Report";
            }
        }

        return [
            'total_scans' => $scans->count(),
            'date_range' => [
                'from' => $filters['dateFrom']->format('Y-m-d'),
                'to' => $filters['dateTo']->format('Y-m-d'),
                'days' => (int) ceil($filters['dateFrom']->diffInDays($filters['dateTo'])),
            ],
            'report_scope' => $scope,
            'generated_by' => auth()->user()->name,
            'generated_at' => now(),
        ];
    }

    /**
     * Generate user analysis (only for admin reports with multiple users)
     */
    private function generateUserAnalysis(Collection $scans, array $filters): ?array
    {
        // Only generate user analysis for admin reports
        if (!auth()->user()->can('viewAny', Scan::class)) {
            return null;
        }

        // Skip if filtering by single user or no users (would be redundant)
        if (!empty($filters['users']) && count($filters['users']) === 1) {
            return null;
        }

        $userStats = $scans->groupBy('user_id')->map(function ($userScans, $userId) {
            $user = $userScans->first()->user;

            return [
                'user' => [
                    'id' => $userId,
                    'name' => $user ? $user->name : 'Unknown User',
                    'email' => $user ? $user->email : 'N/A',
                    'role' => $user ? $user->role : 'N/A',
                ],
                'statistics' => [
                    'total_scans' => $userScans->count(),
                    'defective_scans' => $userScans->where('is_defect', true)->count(),
                    'good_scans' => $userScans->where('is_defect', false)->count(),
                    'defect_rate' => $userScans->count() > 0
                        ? round(($userScans->where('is_defect', true)->count() / $userScans->count()) * 100, 1)
                        : 0,
                    'avg_anomaly_score' => round($userScans->avg('anomaly_score'), 4),
                    'total_defects_found' => $userScans->sum(function ($scan) {
                        return $scan->scanDefects->count();
                    }),
                    'avg_processing_time' => round($userScans->map(function ($scan) {
                        return ($scan->preprocessing_time_ms ?? 0) +
                            ($scan->anomaly_inference_time_ms ?? 0) +
                            ($scan->classification_inference_time_ms ?? 0) +
                            ($scan->postprocessing_time_ms ?? 0);
                    })->avg(), 2),
                ]
            ];
        })->sortByDesc('statistics.total_scans')->values()->toArray();

        return $userStats;
    }

    // Rest of the methods remain unchanged from the original implementation
    private function processDefectsWithExplanations(Collection $scanDefects, Collection $defectTypes): Collection
    {
        return $scanDefects->map(function ($defect) use ($defectTypes) {
            $defectSlug = str()->slug($defect->label);
            $defectType = $defectTypes->get($defectSlug);

            // Handle double-encoded JSON for box_location
            $boxLocation = $defect->box_location;
            if (is_string($boxLocation)) {
                // Try to decode if it's a string (might be double-encoded)
                $decoded = json_decode($boxLocation, true);
                if (json_last_error() === JSON_ERROR_NONE) {
                    $boxLocation = $decoded;
                } else {
                    // If first decode fails, try double-decode
                    $doubleDecoded = json_decode(json_decode($boxLocation, true), true);
                    if (json_last_error() === JSON_ERROR_NONE) {
                        $boxLocation = $doubleDecoded;
                    } else {
                        // Fallback to default values
                        $boxLocation = ['x' => 0, 'y' => 0, 'width' => 0, 'height' => 0];
                    }
                }
            }

            return [
                'id' => $defect->id,
                'label' => $defect->label,
                'confidence_score' => $defect->confidence_score,
                'severity_level' => $defect->severity_level,
                'area_percentage' => $defect->area_percentage,
                'box_location' => $boxLocation,
                'explanation' => $defectType ? $defectType->description : 'No detailed explanation available for this defect type.',
                'defect_type_found' => (bool) $defectType,
                'created_at' => $defect->created_at,
            ];
        });
    }

    private function generateSingleScanCharts(Scan $scan): array
    {
        $defects = $scan->scanDefects;

        // Defect distribution by type
        $defectDistribution = $defects->groupBy('label')->map(function ($group) {
            return [
                'label' => $group->first()->label,
                'count' => $group->count(),
                'avg_confidence' => round($group->avg('confidence_score'), 4),
                'total_area' => round($group->sum('area_percentage'), 2),
            ];
        })->values()->toArray();

        // Severity distribution
        $severityDistribution = $defects->groupBy('severity_level')->map(function ($group, $severity) use ($defects) {
            return [
                'severity' => $severity ?: 'Unknown',
                'count' => $group->count(),
                'percentage' => $defects->count() > 0 ? round(($group->count() / $defects->count()) * 100, 1) : 0,
            ];
        })->values()->toArray();

        // Processing time breakdown
        $processingTimes = [
            'preprocessing' => $scan->preprocessing_time_ms ?? 0,
            'anomaly_inference' => $scan->anomaly_inference_time_ms ?? 0,
            'classification_inference' => $scan->classification_inference_time_ms ?? 0,
            'postprocessing' => $scan->postprocessing_time_ms ?? 0,
        ];

        $totalProcessingTime = array_sum(array_filter($processingTimes));

        $processingTimeBreakdown = collect($processingTimes)->map(function ($time, $stage) use ($totalProcessingTime) {
            return [
                'stage' => str_replace('_', ' ', ucwords(str_replace('_', ' ', $stage))),
                'time_ms' => $time,
                'percentage' => $totalProcessingTime > 0 ? round(($time / $totalProcessingTime) * 100, 1) : 0,
            ];
        })->filter(fn($item) => $item['time_ms'] > 0)->values()->toArray();

        return [
            'defect_distribution' => $defectDistribution,
            'severity_distribution' => $severityDistribution,
            'processing_time_breakdown' => $processingTimeBreakdown,
            'total_processing_time' => $totalProcessingTime,
        ];
    }

    private function getImagePaths(Scan $scan): array
    {
        return [
            'original' => Storage::disk('public')->exists($scan->original_path)
                ? Storage::disk('public')->path($scan->original_path)
                : null,
            'annotated' => Storage::disk('public')->exists($scan->annotated_path)
                ? Storage::disk('public')->path($scan->annotated_path)
                : null,
        ];
    }

    private function generateScanSummary(Scan $scan): array
    {
        $defects = $scan->scanDefects;
        $threat = $scan->scanThreat;

        return [
            'overall_status' => $scan->is_defect ? 'DEFECTIVE' : 'GOOD',
            'anomaly_score' => $scan->anomaly_score,
            'confidence_level' => $scan->anomaly_confidence_level,
            'threshold_used' => $scan->anomaly_threshold,
            'defects_found' => $defects->count(),
            'unique_defect_types' => $defects->pluck('label')->unique()->count(),
            'severity_breakdown' => $defects->groupBy('severity_level')->map->count(),
            'total_affected_area' => round($defects->sum('area_percentage'), 2),
            'avg_defect_confidence' => $defects->count() > 0 ? round($defects->avg('confidence_score'), 4) : 0,
            'threat_status' => $threat ? $threat->status : 'not_scanned',
            'risk_level' => $threat ? $threat->risk_level : 'none',
            'scan_date' => $scan->created_at,
            'processing_performance' => [
                'total_time' => ($scan->preprocessing_time_ms ?? 0) +
                    ($scan->anomaly_inference_time_ms ?? 0) +
                    ($scan->classification_inference_time_ms ?? 0) +
                    ($scan->postprocessing_time_ms ?? 0),
                'fastest_stage' => $this->getFastestProcessingStage($scan),
                'slowest_stage' => $this->getSlowestProcessingStage($scan),
            ],
        ];
    }

    private function getFastestProcessingStage(Scan $scan): string
    {
        $times = [
            'preprocessing' => $scan->preprocessing_time_ms ?? PHP_FLOAT_MAX,
            'anomaly_inference' => $scan->anomaly_inference_time_ms ?? PHP_FLOAT_MAX,
            'classification_inference' => $scan->classification_inference_time_ms ?? PHP_FLOAT_MAX,
            'postprocessing' => $scan->postprocessing_time_ms ?? PHP_FLOAT_MAX,
        ];

        return array_search(min($times), $times);
    }

    private function getSlowestProcessingStage(Scan $scan): string
    {
        $times = [
            'preprocessing' => $scan->preprocessing_time_ms ?? 0,
            'anomaly_inference' => $scan->anomaly_inference_time_ms ?? 0,
            'classification_inference' => $scan->classification_inference_time_ms ?? 0,
            'postprocessing' => $scan->postprocessing_time_ms ?? 0,
        ];

        return array_search(max($times), $times);
    }

    private function generateBatchStatistics(Collection $scans, array $filters): array
    {
        $totalScans = $scans->count();
        $defectiveScans = $scans->where('is_defect', true)->count();
        $goodScans = $totalScans - $defectiveScans;

        return [
            'total_scans' => $totalScans,
            'defective_scans' => $defectiveScans,
            'good_scans' => $goodScans,
            'defect_rate' => $totalScans > 0 ? round(($defectiveScans / $totalScans) * 100, 2) : 0,
            'good_rate' => $totalScans > 0 ? round(($goodScans / $totalScans) * 100, 2) : 0,
            'total_defects_found' => $scans->sum(fn($scan) => $scan->scanDefects->count()),
            'avg_anomaly_score' => $scans->avg('anomaly_score'),
            'avg_processing_time' => $this->calculateAverageProcessingTime($scans),
            'date_range' => [
                'from' => $filters['dateFrom']->format('Y-m-d'),
                'to' => $filters['dateTo']->format('Y-m-d'),
                'days' => $filters['dateFrom']->diffInDays($filters['dateTo']) + 1,
            ],
            'unique_users' => $scans->unique('user_id')->count(),
            'scans_by_confidence' => $this->groupScansByConfidence($scans),
            'processing_performance' => $this->analyzeBatchProcessingPerformance($scans),
            'threat_analysis' => $this->analyzeBatchThreats($scans),
        ];
    }

    private function generateBatchChartsData(Collection $scans, array $filters): array
    {
        return [
            'daily_scan_trends' => $this->generateDailyScanTrends($scans, $filters),
            'defect_type_distribution' => $this->generateDefectTypeDistribution($scans),
            'severity_distribution' => $this->generateSeverityDistribution($scans),
            'confidence_level_distribution' => $this->generateConfidenceLevelDistribution($scans),
            'processing_time_analysis' => $this->generateProcessingTimeAnalysis($scans),
            'anomaly_score_distribution' => $this->generateAnomalyScoreDistribution($scans),
            'user_activity_analysis' => $this->generateUserActivityAnalysis($scans),
            'hourly_scan_patterns' => $this->generateHourlyScanPatterns($scans),
            'threat_status_distribution' => $this->generateThreatStatusDistribution($scans),
        ];
    }

    private function generateBatchDefectAnalysis(Collection $scans, Collection $defectTypes): array
    {
        $allDefects = $scans->flatMap->scanDefects;

        return $allDefects->groupBy('label')->map(function ($defects, $label) use ($defectTypes) {
            $defectSlug = str()->slug($label);
            $defectType = $defectTypes->get($defectSlug);

            return [
                'label' => $label,
                'total_occurrences' => $defects->count(),
                'avg_confidence' => round($defects->avg('confidence_score'), 4),
                'avg_area_affected' => round($defects->avg('area_percentage'), 2),
                'severity_breakdown' => $defects->groupBy('severity_level')->map->count()->toArray(),
                'explanation' => $defectType ? $defectType->description : 'No detailed explanation available.',
                'scans_affected' => $defects->pluck('scan_id')->unique()->count(),
            ];
        })->sortByDesc('total_occurrences')->values()->toArray();
    }

    // Helper methods for batch statistics
    private function calculateAverageProcessingTime(Collection $scans): float
    {
        $totalTimes = $scans->map(function ($scan) {
            return ($scan->preprocessing_time_ms ?? 0) +
                ($scan->anomaly_inference_time_ms ?? 0) +
                ($scan->classification_inference_time_ms ?? 0) +
                ($scan->postprocessing_time_ms ?? 0);
        })->filter();

        return round($totalTimes->avg(), 2);
    }

    private function groupScansByConfidence(Collection $scans): array
    {
        return $scans->groupBy('anomaly_confidence_level')->map(function ($group, $level) {
            return [
                'level' => $level ?: 'Unknown',
                'count' => $group->count(),
                'defect_rate' => $group->count() > 0 ? round(($group->where('is_defect', true)->count() / $group->count()) * 100, 1) : 0,
            ];
        })->toArray();
    }

    private function analyzeBatchProcessingPerformance(Collection $scans): array
    {
        $totalTimes = $scans->map(function ($scan) {
            return ($scan->preprocessing_time_ms ?? 0) +
                ($scan->anomaly_inference_time_ms ?? 0) +
                ($scan->classification_inference_time_ms ?? 0) +
                ($scan->postprocessing_time_ms ?? 0);
        })->filter();

        return [
            'average_total_time' => round($totalTimes->avg(), 2),
            'fastest_scan' => round($totalTimes->min(), 2),
            'slowest_scan' => round($totalTimes->max(), 2),
            'total_processing_time' => round($totalTimes->sum(), 2),
            'scans_per_minute' => $totalTimes->count() > 0 ? round(60000 / $totalTimes->avg(), 2) : 0,
        ];
    }

    private function analyzeBatchThreats(Collection $scans): array
    {
        $threats = $scans->pluck('scanThreat')->filter();
        $totalThreats = $threats->count();

        if ($totalThreats === 0) {
            return [
                'total_threats_scanned' => 0,
                'clean_count' => 0,
                'suspicious_count' => 0,
                'malicious_count' => 0,
                'clean_percentage' => 0,
                'risk_distribution' => [],
            ];
        }

        return [
            'total_threats_scanned' => $totalThreats,
            'clean_count' => $threats->where('status', 'clean')->count(),
            'suspicious_count' => $threats->where('status', 'suspicious')->count(),
            'malicious_count' => $threats->where('status', 'malicious')->count(),
            'clean_percentage' => round(($threats->where('status', 'clean')->count() / $totalThreats) * 100, 1),
            'risk_distribution' => $threats->groupBy('risk_level')->map->count()->toArray(),
        ];
    }

    private function generateDailyScanTrends(Collection $scans, array $filters): array
    {
        return $scans->groupBy(function ($scan) {
            return $scan->created_at->format('Y-m-d');
        })->map(function ($dayScans, $date) {
            return [
                'date' => $date,
                'total' => $dayScans->count(),
                'defective' => $dayScans->where('is_defect', true)->count(),
                'good' => $dayScans->where('is_defect', false)->count(),
            ];
        })->sortBy('date')->values()->toArray();
    }

    private function generateDefectTypeDistribution(Collection $scans): array
    {
        $allDefects = $scans->flatMap->scanDefects;

        return $allDefects->groupBy('label')->map(function ($defects, $label) {
            return [
                'label' => $label,
                'count' => $defects->count(),
                'avg_confidence' => round($defects->avg('confidence_score'), 4),
                'total_area' => round($defects->sum('area_percentage'), 2),
            ];
        })->sortByDesc('count')->values()->toArray();
    }

    private function generateSeverityDistribution(Collection $scans): array
    {
        $allDefects = $scans->flatMap->scanDefects;
        $totalDefects = $allDefects->count();

        return $allDefects->groupBy('severity_level')->map(function ($defects, $severity) use ($totalDefects) {
            return [
                'severity' => $severity ?: 'Unknown',
                'count' => $defects->count(),
                'percentage' => $totalDefects > 0 ? round(($defects->count() / $totalDefects) * 100, 1) : 0,
            ];
        })->values()->toArray();
    }

    private function generateConfidenceLevelDistribution(Collection $scans): array
    {
        return $scans->groupBy('anomaly_confidence_level')->map(function ($scanGroup, $level) {
            return [
                'level' => $level ?: 'Unknown',
                'count' => $scanGroup->count(),
                'avg_anomaly_score' => round($scanGroup->avg('anomaly_score'), 4),
            ];
        })->values()->toArray();
    }

    private function generateProcessingTimeAnalysis(Collection $scans): array
    {
        $processingStages = [
            'preprocessing' => 'preprocessing_time_ms',
            'anomaly_inference' => 'anomaly_inference_time_ms',
            'classification_inference' => 'classification_inference_time_ms',
            'postprocessing' => 'postprocessing_time_ms',
        ];

        return collect($processingStages)->map(function ($field, $stage) use ($scans) {
            $times = $scans->pluck($field)->filter();
            return [
                'stage' => str_replace('_', ' ', ucwords(str_replace('_', ' ', $stage))),
                'avg_time' => round($times->avg(), 2),
                'min_time' => round($times->min(), 2),
                'max_time' => round($times->max(), 2),
                'total_time' => round($times->sum(), 2),
            ];
        })->values()->toArray();
    }

    private function generateAnomalyScoreDistribution(Collection $scans): array
    {
        $ranges = [
            '0.0-0.2' => [0.0, 0.2],
            '0.2-0.4' => [0.2, 0.4],
            '0.4-0.6' => [0.4, 0.6],
            '0.6-0.8' => [0.6, 0.8],
            '0.8-1.0' => [0.8, 1.0],
        ];

        return collect($ranges)->map(function ($range, $label) use ($scans) {
            $count = $scans->whereBetween('anomaly_score', $range)->count();
            return [
                'range' => $label,
                'count' => $count,
                'percentage' => $scans->count() > 0 ? round(($count / $scans->count()) * 100, 1) : 0,
            ];
        })->values()->toArray();
    }

    private function generateUserActivityAnalysis(Collection $scans): array
    {
        return $scans->groupBy('user_id')->map(function ($userScans, $userId) {
            $user = $userScans->first()->user;
            return [
                'user_name' => $user ? $user->name : 'Unknown User',
                'user_email' => $user ? $user->email : 'N/A',
                'total_scans' => $userScans->count(),
                'defective_scans' => $userScans->where('is_defect', true)->count(),
                'avg_anomaly_score' => round($userScans->avg('anomaly_score'), 4),
                'defect_rate' => $userScans->count() > 0 ? round(($userScans->where('is_defect', true)->count() / $userScans->count()) * 100, 1) : 0,
            ];
        })->sortByDesc('total_scans')->values()->toArray();
    }

    private function generateHourlyScanPatterns(Collection $scans): array
    {
        return $scans->groupBy(function ($scan) {
            return $scan->created_at->format('H');
        })->map(function ($hourScans, $hour) {
            return [
                'hour' => sprintf('%02d:00', $hour),
                'total' => $hourScans->count(),
                'defective' => $hourScans->where('is_defect', true)->count(),
                'good' => $hourScans->where('is_defect', false)->count(),
            ];
        })->sortBy('hour')->values()->toArray();
    }

    private function generateThreatStatusDistribution(Collection $scans): array
    {
        $threatsData = $scans->pluck('scanThreat')->filter();

        if ($threatsData->isEmpty()) {
            return [];
        }

        return $threatsData->groupBy('status')->map(function ($threats, $status) {
            return [
                'status' => $status,
                'count' => $threats->count(),
                'risk_levels' => $threats->groupBy('risk_level')->map->count()->toArray(),
            ];
        })->values()->toArray();
    }
}
