<?php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

class ChartGenerationService
{
    private int $timeoutSeconds = 30;
    private string $baseUrl = 'https://quickchart.io/chart';

    /**
     * Generate chart as base64 image using QuickChart API
     */
    public function generateChartBase64(array $chartConfig, int $width = 400, int $height = 200): ?string
    {
        try {
            Log::info('Generating chart', [
                'chart_type' => $chartConfig['type'] ?? 'unknown',
                'width' => $width,
                'height' => $height
            ]);

            $response = Http::timeout($this->timeoutSeconds)->post($this->baseUrl, [
                'chart' => $chartConfig,
                'width' => $width,
                'height' => $height,
                'format' => 'png',
                'backgroundColor' => 'white',
            ]);

            if ($response->successful()) {
                Log::info('Chart generated successfully', [
                    'chart_type' => $chartConfig['type'] ?? 'unknown',
                    'response_size' => strlen($response->body())
                ]);
                return 'data:image/png;base64,' . base64_encode($response->body());
            }

            Log::error('Chart generation failed - API error', [
                'status' => $response->status(),
                'body' => $response->body(),
                'chart_type' => $chartConfig['type'] ?? 'unknown'
            ]);

            // Return placeholder on API failure
            return $this->generatePlaceholderChart('Chart Generation Failed', 'API temporarily unavailable');
        } catch (\Exception $e) {
            Log::error('Chart generation exception', [
                'error' => $e->getMessage(),
                'chart_type' => $chartConfig['type'] ?? 'unknown',
                'trace' => $e->getTraceAsString()
            ]);

            // Return placeholder on exception
            return $this->generatePlaceholderChart('Chart Generation Error', $e->getMessage());
        }
    }

    /**
     * Generate defect distribution doughnut chart
     */
    public function generateDefectDistributionChart(array $defectData): ?string
    {
        if (empty($defectData)) {
            Log::warning('No defect data provided for chart generation');
            return $this->generatePlaceholderChart('No Defect Data', 'No defects found to display');
        }

        $labels = array_column($defectData, 'label');
        $data = array_column($defectData, 'count');
        $colors = ['#dc3545', '#ffc107', '#28a745', '#007bff', '#6f42c1', '#fd7e14', '#20c997', '#e83e8c'];

        $chartConfig = [
            'type' => 'doughnut',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 2,
                    'borderColor' => '#fff'
                ]]
            ],
            'options' => [
                'responsive' => false,
                'plugins' => [
                    'legend' => [
                        'position' => 'bottom',
                        'labels' => [
                            'fontSize' => 12,
                            'padding' => 15
                        ]
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Defect Type Distribution'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 350, 250);
    }

    /**
     * Generate severity distribution pie chart
     */
    public function generateSeverityDistributionChart(array $severityData): ?string
    {
        if (empty($severityData)) {
            return $this->generatePlaceholderChart('No Severity Data', 'No severity levels to display');
        }

        $labels = array_column($severityData, 'severity');
        $data = array_column($severityData, 'count');
        $colors = ['#dc3545', '#ffc107', '#28a745']; // Red, Yellow, Green

        $chartConfig = [
            'type' => 'pie',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 2,
                    'borderColor' => '#fff'
                ]]
            ],
            'options' => [
                'responsive' => false,
                'plugins' => [
                    'legend' => [
                        'position' => 'bottom',
                        'labels' => [
                            'fontSize' => 12,
                            'padding' => 15
                        ]
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Severity Level Distribution'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 350, 250);
    }

    /**
     * Generate processing time bar chart
     */
    public function generateProcessingTimeChart(array $processingData): ?string
    {
        if (empty($processingData)) {
            return $this->generatePlaceholderChart('No Processing Data', 'No processing times to display');
        }

        $labels = array_column($processingData, 'stage');
        $data = array_column($processingData, 'time_ms');
        $colors = ['#007bff', '#28a745', '#ffc107', '#6f42c1'];

        $chartConfig = [
            'type' => 'bar',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'label' => 'Time (ms)',
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 1,
                    'borderColor' => '#fff'
                ]]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Time (milliseconds)'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'display' => false
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Processing Time Breakdown'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 600, 300);
    }

    /**
     * Generate daily scan trends line chart
     */
    public function generateDailyScanTrendsChart(array $trendsData): ?string
    {
        if (empty($trendsData)) {
            return $this->generatePlaceholderChart('No Trend Data', 'No daily trends to display');
        }

        $labels = array_column($trendsData, 'date');
        $totalData = array_column($trendsData, 'total');
        $defectiveData = array_column($trendsData, 'defective');
        $goodData = array_column($trendsData, 'good');

        $chartConfig = [
            'type' => 'line',
            'data' => [
                'labels' => $labels,
                'datasets' => [
                    [
                        'label' => 'Total Scans',
                        'data' => $totalData,
                        'borderColor' => '#007bff',
                        'backgroundColor' => 'rgba(0, 123, 255, 0.1)',
                        'borderWidth' => 2,
                        'fill' => true,
                        'tension' => 0.1
                    ],
                    [
                        'label' => 'Defective',
                        'data' => $defectiveData,
                        'borderColor' => '#dc3545',
                        'backgroundColor' => 'rgba(220, 53, 69, 0.1)',
                        'borderWidth' => 2,
                        'fill' => false,
                        'tension' => 0.1
                    ],
                    [
                        'label' => 'Good',
                        'data' => $goodData,
                        'borderColor' => '#28a745',
                        'backgroundColor' => 'rgba(40, 167, 69, 0.1)',
                        'borderWidth' => 2,
                        'fill' => false,
                        'tension' => 0.1
                    ]
                ]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Number of Scans'
                        ]
                    ],
                    'x' => [
                        'title' => [
                            'display' => true,
                            'text' => 'Date'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'position' => 'top'
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Daily Scan Trends'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 700, 400);
    }

    /**
     * Generate confidence level distribution chart
     */
    public function generateConfidenceLevelChart(array $confidenceData): ?string
    {
        if (empty($confidenceData)) {
            return $this->generatePlaceholderChart('No Confidence Data', 'No confidence levels to display');
        }

        $labels = array_column($confidenceData, 'level');
        $data = array_column($confidenceData, 'count');
        $colors = ['#28a745', '#ffc107', '#dc3545', '#007bff'];

        $chartConfig = [
            'type' => 'bar',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'label' => 'Scan Count',
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 1
                ]]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Number of Scans'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'display' => false
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Confidence Level Distribution'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 500, 300);
    }

    /**
     * Generate anomaly score distribution chart
     */
    public function generateAnomalyScoreChart(array $anomalyData): ?string
    {
        if (empty($anomalyData)) {
            return $this->generatePlaceholderChart('No Anomaly Data', 'No anomaly scores to display');
        }

        $labels = array_column($anomalyData, 'range');
        $data = array_column($anomalyData, 'count');
        $colors = ['#28a745', '#20c997', '#ffc107', '#fd7e14', '#dc3545'];

        $chartConfig = [
            'type' => 'bar',
            'data' => [
                'labels' => $labels,
                'datasets' => [[
                    'label' => 'Scan Count',
                    'data' => $data,
                    'backgroundColor' => array_slice($colors, 0, count($data)),
                    'borderWidth' => 1
                ]]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Number of Scans'
                        ]
                    ],
                    'x' => [
                        'title' => [
                            'display' => true,
                            'text' => 'Anomaly Score Range'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'display' => false
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Anomaly Score Distribution'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 600, 350);
    }

    /**
     * Generate user activity chart
     */
    public function generateUserActivityChart(array $userData): ?string
    {
        if (empty($userData)) {
            return $this->generatePlaceholderChart('No User Data', 'No user activity to display');
        }

        // Limit to top 10 users to avoid overcrowding
        $topUsers = array_slice($userData, 0, 10);

        $labels = array_column($topUsers, 'user_name');
        $totalScans = array_column($topUsers, 'total_scans');
        $defectiveScans = array_column($topUsers, 'defective_scans');

        $chartConfig = [
            'type' => 'bar',
            'data' => [
                'labels' => $labels,
                'datasets' => [
                    [
                        'label' => 'Total Scans',
                        'data' => $totalScans,
                        'backgroundColor' => '#007bff',
                        'borderWidth' => 1
                    ],
                    [
                        'label' => 'Defective Scans',
                        'data' => $defectiveScans,
                        'backgroundColor' => '#dc3545',
                        'borderWidth' => 1
                    ]
                ]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Number of Scans'
                        ]
                    ],
                    'x' => [
                        'title' => [
                            'display' => true,
                            'text' => 'Users'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'position' => 'top'
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'User Activity Comparison'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 700, 400);
    }

    /**
     * Generate hourly patterns chart
     */
    public function generateHourlyPatternsChart(array $hourlyData): ?string
    {
        if (empty($hourlyData)) {
            return $this->generatePlaceholderChart('No Hourly Data', 'No hourly patterns to display');
        }

        $labels = array_column($hourlyData, 'hour');
        $totalData = array_column($hourlyData, 'total');
        $defectiveData = array_column($hourlyData, 'defective');

        $chartConfig = [
            'type' => 'line',
            'data' => [
                'labels' => $labels,
                'datasets' => [
                    [
                        'label' => 'Total Scans',
                        'data' => $totalData,
                        'borderColor' => '#007bff',
                        'backgroundColor' => 'rgba(0, 123, 255, 0.1)',
                        'borderWidth' => 2,
                        'fill' => true,
                        'tension' => 0.1
                    ],
                    [
                        'label' => 'Defective Scans',
                        'data' => $defectiveData,
                        'borderColor' => '#dc3545',
                        'backgroundColor' => 'rgba(220, 53, 69, 0.1)',
                        'borderWidth' => 2,
                        'fill' => false,
                        'tension' => 0.1
                    ]
                ]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Number of Scans'
                        ]
                    ],
                    'x' => [
                        'title' => [
                            'display' => true,
                            'text' => 'Hour of Day'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'position' => 'top'
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Hourly Scan Patterns'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 700, 350);
    }

    /**
     * Generate processing performance comparison chart
     */
    public function generateProcessingPerformanceChart(array $performanceData): ?string
    {
        if (empty($performanceData)) {
            return $this->generatePlaceholderChart('No Performance Data', 'No processing performance to display');
        }

        $labels = array_column($performanceData, 'stage');
        $avgTime = array_column($performanceData, 'avg_time');
        $maxTime = array_column($performanceData, 'max_time');

        $chartConfig = [
            'type' => 'bar',
            'data' => [
                'labels' => $labels,
                'datasets' => [
                    [
                        'label' => 'Average Time (ms)',
                        'data' => $avgTime,
                        'backgroundColor' => '#007bff',
                        'borderWidth' => 1
                    ],
                    [
                        'label' => 'Max Time (ms)',
                        'data' => $maxTime,
                        'backgroundColor' => '#dc3545',
                        'borderWidth' => 1
                    ]
                ]
            ],
            'options' => [
                'responsive' => false,
                'scales' => [
                    'y' => [
                        'beginAtZero' => true,
                        'title' => [
                            'display' => true,
                            'text' => 'Time (milliseconds)'
                        ]
                    ]
                ],
                'plugins' => [
                    'legend' => [
                        'position' => 'top'
                    ],
                    'title' => [
                        'display' => true,
                        'text' => 'Processing Performance Comparison'
                    ]
                ]
            ]
        ];

        return $this->generateChartBase64($chartConfig, 600, 350);
    }

    /**
     * Generate fallback placeholder image
     */
    public function generatePlaceholderChart(string $title, string $subtitle = ''): string
    {
        // Create a simple SVG placeholder and convert to base64
        $svg = '<svg width="350" height="250" xmlns="http://www.w3.org/2000/svg">
            <rect width="100%" height="100%" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2" rx="8"/>
            <circle cx="175" cy="90" r="30" fill="#e9ecef" stroke="#dee2e6" stroke-width="2"/>
            <rect x="155" y="70" width="40" height="10" fill="#dee2e6" rx="2"/>
            <rect x="165" y="85" width="20" height="5" fill="#dee2e6" rx="1"/>
            <rect x="155" y="95" width="40" height="5" fill="#dee2e6" rx="1"/>
            <text x="50%" y="65%" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#495057" font-weight="bold">' . htmlspecialchars($title) . '</text>
            <text x="50%" y="75%" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#6c757d">' . htmlspecialchars($subtitle) . '</text>
        </svg>';

        return 'data:image/svg+xml;base64,' . base64_encode($svg);
    }

    /**
     * Set timeout for chart generation requests
     */
    public function setTimeout(int $seconds): self
    {
        $this->timeoutSeconds = $seconds;
        return $this;
    }

    /**
     * Set base URL for chart generation API
     */
    public function setBaseUrl(string $url): self
    {
        $this->baseUrl = rtrim($url, '/');
        return $this;
    }
}
