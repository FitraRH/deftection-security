<?php

namespace App\Services;

use App\Models\Scan;
use App\Models\DefectType;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Log;

class ScanDetailService
{
    /**
     * Transform scan data for the analysis detail page
     */
    public function transformScanForAnalysis(Scan $scan): array
    {
        try {
            Log::info('Starting scan transformation', [
                'scan_id' => $scan->id,
                'filename' => $scan->filename,
                'is_defect' => $scan->is_defect,
            ]);

            // Load relationships with error handling
            $this->loadScanRelationships($scan);

            // Calculate derived values with validation
            $calculations = $this->performCalculations($scan);

            // Get image URLs with validation
            $imageUrls = $this->getImageUrls($scan);

            // Process defects data with explanations
            $defects = $this->transformDefectsWithExplanations($scan->scanDefects);

            $result = [
                'id' => $scan->id,
                'status' => $calculations['status'],
                'summary' => $this->buildSummary($scan, $calculations),
                'visuals' => $this->buildVisuals($calculations['status'], $imageUrls),
                'defects' => $defects,
                'performance' => $this->buildPerformance($scan, $calculations),
                'technical' => $this->buildTechnical($scan, $calculations, $defects),
            ];

            Log::info('Scan transformation completed successfully', [
                'scan_id' => $scan->id,
                'status' => $result['status'],
                'defects_count' => count($defects),
                'has_threat' => !is_null($scan->scanThreat),
            ]);

            return $result;
        } catch (\Exception $e) {
            Log::error('Error during scan transformation', [
                'scan_id' => $scan->id,
                'error' => $e->getMessage(),
                'trace' => $e->getTraceAsString(),
            ]);

            throw $e;
        }
    }

    /**
     * Load scan relationships with error handling
     */
    private function loadScanRelationships(Scan $scan): void
    {
        try {
            $scan->load(['scanDefects', 'scanThreat', 'user']);

            Log::debug('Scan relationships loaded', [
                'scan_id' => $scan->id,
                'defects_count' => $scan->scanDefects->count(),
                'has_threat' => !is_null($scan->scanThreat),
                'has_user' => !is_null($scan->user),
            ]);
        } catch (\Exception $e) {
            Log::warning('Failed to load some scan relationships', [
                'scan_id' => $scan->id,
                'error' => $e->getMessage(),
            ]);
        }
    }

    /**
     * Perform all calculations needed for the analysis
     */
    private function performCalculations(Scan $scan): array
    {
        $totalProcessingTime = $this->calculateTotalProcessingTime($scan);
        $aiInferenceTime = $this->calculateAiInferenceTime($scan);
        $status = $scan->is_defect ? 'defect' : 'good';

        // Use database anomaly_confidence_level if available, otherwise calculate
        $anomalyConfidenceLevel = $scan->anomaly_confidence_level ?? $this->getConfidenceLevel($scan->anomaly_score ?? 0);

        return [
            'totalProcessingTime' => $totalProcessingTime,
            'aiInferenceTime' => $aiInferenceTime,
            'status' => $status,
            'anomalyConfidenceLevel' => $anomalyConfidenceLevel,
        ];
    }

    /**
     * Calculate total processing time in milliseconds with null safety
     */
    private function calculateTotalProcessingTime(Scan $scan): float
    {
        $total = ($scan->preprocessing_time_ms ?? 0) +
            ($scan->anomaly_inference_time_ms ?? 0) +
            ($scan->classification_inference_time_ms ?? 0) +
            ($scan->postprocessing_time_ms ?? 0);

        return max(0, $total);
    }

    /**
     * Calculate AI inference time (anomaly + classification) with null safety
     */
    private function calculateAiInferenceTime(Scan $scan): float
    {
        $total = ($scan->anomaly_inference_time_ms ?? 0) +
            ($scan->classification_inference_time_ms ?? 0);

        return max(0, $total);
    }

    /**
     * Determine confidence level based on anomaly score (fallback method)
     */
    private function getConfidenceLevel(?float $anomalyScore): string
    {
        $score = $anomalyScore ?? 0;

        return match (true) {
            $score >= 0.8 => 'Very High',
            $score >= 0.7 => 'High',
            $score >= 0.5 => 'Medium',
            $score >= 0.3 => 'Low',
            default => 'Very Low'
        };
    }

    /**
     * Get image URLs with error handling and validation
     */
    private function getImageUrls(Scan $scan): array
    {
        $originalImageUrl = '';
        $analyzedImageUrl = '';

        try {
            if ($scan->original_path && Storage::disk('public')->exists($scan->original_path)) {
                $originalImageUrl = Storage::url($scan->original_path);
            }

            if ($scan->annotated_path && Storage::disk('public')->exists($scan->annotated_path)) {
                $analyzedImageUrl = Storage::url($scan->annotated_path);
            }
        } catch (\Exception $e) {
            Log::error('Error processing image URLs', [
                'scan_id' => $scan->id,
                'error' => $e->getMessage(),
            ]);
        }

        return [
            'original' => $originalImageUrl,
            'analyzed' => $analyzedImageUrl,
        ];
    }

    /**
     * Transform defects collection with explanations from DefectType table
     */
    private function transformDefectsWithExplanations($scanDefects): array
    {
        try {
            if (!$scanDefects || $scanDefects->isEmpty()) {
                return [];
            }

            // Get all defect types for explanation lookup
            $defectTypes = DefectType::all()->keyBy('slug');

            $defects = $scanDefects->map(function ($defect) use ($defectTypes) {
                try {
                    // Format label and find matching defect type
                    $formattedLabel = $this->formatDefectLabel($defect->label ?? 'Unknown');
                    $defectTypeSlug = str_replace(' ', '_', strtolower($defect->label ?? ''));

                    // Look for explanation in DefectType table
                    $explanation = null;
                    if (isset($defectTypes[$defectTypeSlug])) {
                        $explanation = $defectTypes[$defectTypeSlug]->description;
                    } else {
                        // Try alternative matching methods
                        foreach ($defectTypes as $defectType) {
                            if (
                                strtolower($defectType->name) === strtolower($formattedLabel) ||
                                strtolower($defectType->slug) === strtolower($defect->label)
                            ) {
                                $explanation = $defectType->description;
                                break;
                            }
                        }
                    }

                    return [
                        'name' => $formattedLabel,
                        'confidence' => round(($defect->confidence_score ?? 0) * 100),
                        'coverage' => round($defect->area_percentage ?? 0, 1),
                        'severity' => $defect->severity_level ?? 'Unknown',
                        'location' => $defect->box_location ?? null,
                        'explanation' => $explanation ?? 'No detailed explanation available for this defect type.',
                        'regions' => 1
                    ];
                } catch (\Exception $e) {
                    Log::warning('Error transforming individual defect', [
                        'defect_id' => $defect->id ?? 'unknown',
                        'error' => $e->getMessage(),
                    ]);

                    return [
                        'name' => 'Unknown Defect',
                        'confidence' => 0,
                        'coverage' => 0,
                        'severity' => 'Unknown',
                        'location' => null,
                        'explanation' => 'Error processing defect information.',
                        'regions' => 1
                    ];
                }
            })->toArray();

            return $defects;
        } catch (\Exception $e) {
            Log::error('Error transforming defects collection', [
                'error' => $e->getMessage(),
            ]);
            return [];
        }
    }

    /**
     * Format defect label for display
     */
    private function formatDefectLabel(string $label): string
    {
        return ucwords(str_replace('_', ' ', $label));
    }

    /**
     * Get image dimensions from stored file
     */
    private function getImageDimensions(?string $imagePath): string
    {
        try {
            if (!$imagePath) return 'Unknown';

            $fullPath = Storage::disk('public')->path($imagePath);
            if (!file_exists($fullPath)) return 'Unknown';

            $imageInfo = getimagesize($fullPath);
            if (!$imageInfo || !isset($imageInfo[0]) || !isset($imageInfo[1])) {
                return 'Unknown';
            }

            return $imageInfo[0] . 'x' . $imageInfo[1];
        } catch (\Exception $e) {
            Log::warning('Error getting image dimensions', [
                'path' => $imagePath,
                'error' => $e->getMessage(),
            ]);
            return 'Unknown';
        }
    }

    /**
     * Build summary section data (updated structure)
     */
    private function buildSummary(Scan $scan, array $calculations): array
    {
        try {
            return [
                'imageName' => $scan->filename ?? 'Unknown',
                'originalSize' => $this->getImageDimensions($scan->original_path),
                'analysisDate' => $scan->created_at ? $scan->created_at->format('d/m/Y, H:i:s') : 'Unknown',
                'scannedBy' => $scan->user ? $scan->user->name . ' (' . $scan->user->email . ')' : 'Unknown User',
                'finalDecision' => $scan->is_defect ? 'DEFECT' : 'GOOD',
                'anomalyScore' => round($scan->anomaly_score ?? 0, 4),
                'anomalyConfidenceLevel' => $calculations['anomalyConfidenceLevel'], // This is anomaly confidence, not AI confidence
                'status' => 'Completed',
                // Processing time information moved to performance card
                'totalProcessingTime' => number_format($calculations['totalProcessingTime'] / 1000, 3) . 's',
                'preprocessingTime' => number_format(($scan->preprocessing_time_ms ?? 0) / 1000, 3) . 's',
                'anomalyInferenceTime' => number_format(($scan->anomaly_inference_time_ms ?? 0) / 1000, 3) . 's',
                'classificationInferenceTime' => number_format(($scan->classification_inference_time_ms ?? 0) / 1000, 3) . 's',
                'postprocessingTime' => number_format(($scan->postprocessing_time_ms ?? 0) / 1000, 3) . 's',
            ];
        } catch (\Exception $e) {
            Log::error('Error building summary', [
                'scan_id' => $scan->id,
                'error' => $e->getMessage(),
            ]);
            throw $e;
        }
    }

    /**
     * Build visuals section data
     */
    private function buildVisuals(string $status, array $imageUrls): array
    {
        return [
            'status' => $status,
            'originalImageUrl' => $imageUrls['original'],
            'analyzedImageUrl' => $status === 'defect' ? $imageUrls['analyzed'] : '',
        ];
    }

    /**
     * Build performance section data (updated structure)
     */
    private function buildPerformance(Scan $scan, array $calculations): array
    {
        try {
            return [
                'timeBreakdown' => [
                    'preprocessing' => round(($scan->preprocessing_time_ms ?? 0) / 1000, 3),
                    'anomalyInference' => round(($scan->anomaly_inference_time_ms ?? 0) / 1000, 3),
                    'classificationInference' => round(($scan->classification_inference_time_ms ?? 0) / 1000, 3),
                    'postprocessing' => round(($scan->postprocessing_time_ms ?? 0) / 1000, 3),
                ],
                'isDefect' => $scan->is_defect, // To determine if we show classification inference
            ];
        } catch (\Exception $e) {
            Log::error('Error building performance data', [
                'scan_id' => $scan->id,
                'error' => $e->getMessage(),
            ]);

            return [
                'timeBreakdown' => [
                    'preprocessing' => 0,
                    'anomalyInference' => 0,
                    'classificationInference' => 0,
                    'postprocessing' => 0,
                ],
                'isDefect' => false,
            ];
        }
    }

    /**
     * Build technical section data (updated structure)
     */
    private function buildTechnical(Scan $scan, array $calculations, array $defects): array
    {
        try {
            // Build parameters (classification only for defects)
            $parameters = [
                'Total Processing Time' => number_format($calculations['totalProcessingTime'] / 1000, 3) . 's',
                'Image Preprocessing' => number_format(($scan->preprocessing_time_ms ?? 0) / 1000, 3) . 's',
                'Anomaly Inference' => number_format(($scan->anomaly_inference_time_ms ?? 0) / 1000, 3) . 's',
                'Postprocessing' => number_format(($scan->postprocessing_time_ms ?? 0) / 1000, 3) . 's',
                'Model Used' => config('app.ai_model_name', 'HRNet + Anomalib'),
            ];

            if ($scan->is_defect) {
                // Insert classification inference before postprocessing
                $tempParams = $parameters;
                unset($tempParams['Postprocessing']);
                $tempParams['Classification Inference'] = number_format(($scan->classification_inference_time_ms ?? 0) / 1000, 3) . 's';
                $tempParams['Postprocessing'] = $parameters['Postprocessing'];
                $parameters = $tempParams;
            }

            // Build scan metrics (updated structure)
            $scanMetrics = [
                'Anomaly Score' => $scan->anomaly_score ?? 0,
                'Anomaly Confidence Level' => $calculations['anomalyConfidenceLevel'],
                'Anomaly Threshold' => $scan->anomaly_threshold ?? 0,
            ];

            // Add classification confidence only for defects
            if ($scan->is_defect && $scan->scanDefects->count() > 0) {
                $avgClassificationConfidence = $scan->scanDefects->avg('confidence_score');
                $scanMetrics['Classification Avg Confidence'] = $avgClassificationConfidence ?? 0;
            }

            // Process threat data if available
            $threatData = null;
            $hasThreat = false;

            if ($scan->scanThreat) {
                $hasThreat = true;
                $threatData = [
                    'status' => strtoupper($scan->scanThreat->status ?? 'unknown'),
                    'riskLevel' => strtoupper($scan->scanThreat->risk_level ?? 'unknown'),
                    'fileHash' => $scan->scanThreat->hash ?? 'N/A',
                    'scanTime' => number_format(($scan->scanThreat->processing_time_ms ?? 0), 1) . 'ms',
                    'securityFlags' => $scan->scanThreat->flags ?? [],
                    'detailedAnalysis' => $scan->scanThreat->details ?? [],
                    'possibleAttacks' => $scan->scanThreat->possible_attack ?? [],
                ];
            }

            return [
                'parameters' => $parameters,
                'scanMetrics' => $scanMetrics,
                'threatData' => $threatData,
                'isDefect' => $scan->is_defect,
                'hasThreat' => $hasThreat,
                'rawData' => [
                    'analysis_date' => $scan->created_at ? $scan->created_at->format('Y-m-d H:i:s') : null,
                    'anomaly_score' => $scan->anomaly_score ?? 0,
                    'anomaly_confidence_level' => $calculations['anomalyConfidenceLevel'],
                    'anomaly_threshold' => $scan->anomaly_threshold ?? 0,
                    'final_decision' => $scan->is_defect ? 'DEFECT' : 'GOOD',
                    'id' => $scan->id,
                    'image_name' => $scan->filename ?? 'Unknown',
                    'defects' => collect($defects)->pluck('name')->toArray(),
                    'threat_status' => $scan->scanThreat?->status ?? null,
                ],
            ];
        } catch (\Exception $e) {
            Log::error('Error building technical data', [
                'scan_id' => $scan->id,
                'error' => $e->getMessage(),
            ]);
            throw $e;
        }
    }
}
