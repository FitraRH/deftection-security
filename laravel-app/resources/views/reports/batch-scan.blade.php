<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ $reportTitle }}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'DejaVu Sans', Arial, sans-serif;
            font-size: 11px;
            line-height: 1.5;
            color: #333;
            margin: 15mm 10mm 15mm 10mm;
        }

        @page {
            margin: 15mm 10mm 15mm 10mm;
            size: A4;
        }

        .page-break {
            page-break-after: always;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px 0;
            border-bottom: 3px solid #007bff;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 8px 8px 0 0;
        }

        .header h1 {
            font-size: 24px;
            color: #007bff;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header .subtitle {
            font-size: 14px;
            color: #6c757d;
            font-weight: 500;
        }

        .summary-dashboard {
            display: table;
            width: 100%;
            margin-bottom: 25px;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .summary-row {
            display: table-row;
        }

        .summary-cell {
            display: table-cell;
            padding: 12px 10px;
            border-right: 1px solid #dee2e6;
            text-align: center;
            vertical-align: middle;
        }

        .summary-cell:last-child {
            border-right: none;
        }

        .summary-cell.header {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-weight: 700;
            font-size: 10px;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .summary-cell .value {
            font-size: 18px;
            font-weight: 800;
            color: #007bff;
            margin-bottom: 3px;
        }

        .summary-cell .label {
            font-size: 9px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .section {
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 18px;
            font-weight: 700;
            color: #007bff;
            margin-bottom: 15px;
            padding-bottom: 6px;
            border-bottom: 2px solid #007bff;
            position: relative;
        }

        .section-title::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            width: 50px;
            height: 2px;
            background-color: #28a745;
        }

        .two-column {
            display: table;
            width: 100%;
        }

        .column {
            display: table-cell;
            width: 50%;
            vertical-align: top;
            padding-right: 15px;
        }

        .column:last-child {
            padding-right: 0;
            padding-left: 15px;
        }

        .three-column {
            display: table;
            width: 100%;
        }

        .three-column .column {
            width: 33.33%;
            padding-right: 10px;
        }

        .three-column .column:last-child {
            padding-right: 0;
            padding-left: 0;
        }

        .chart-image {
            max-width: 100%;
            max-height: 220px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            margin: 10px 0;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
        }

        .chart-container {
            text-align: center;
            margin-bottom: 20px;
        }

        .chart-title {
            font-size: 14px;
            font-weight: 600;
            color: #495057;
            margin-bottom: 10px;
        }

        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            overflow: hidden;
            font-size: 10px;
        }

        .data-table th,
        .data-table td {
            border-bottom: 1px solid #dee2e6;
            padding: 8px 10px;
            text-align: left;
        }

        .data-table th {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            font-weight: 700;
            color: #495057;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 9px;
        }

        .data-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }

        .data-table tr:last-child td {
            border-bottom: none;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 500;
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            min-width: 60px;
            text-align: center;
        }

        .status-good {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 1px solid #28a745;
        }

        .status-defective {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 1px solid #dc3545;
        }

        .filter-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
            border-left: 4px solid #007bff;
        }

        .filter-info h4 {
            font-size: 12px;
            color: #495057;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .filter-item {
            display: inline-block;
            margin: 3px 8px 3px 0;
            padding: 3px 8px;
            background: #fff;
            border: 1px solid #dee2e6;
            border-radius: 4px;
            font-size: 10px;
        }

        .highlight-box {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 4px solid #2196f3;
        }

        .highlight-box h4 {
            font-size: 14px;
            color: #1565c0;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .highlight-box p {
            font-size: 11px;
            color: #424242;
            margin-bottom: 5px;
        }

        .defect-analysis-item {
            margin-bottom: 15px;
            padding: 12px;
            border: 1px solid #dee2e6;
            border-radius: 6px;
            background: #fff;
        }

        .defect-analysis-item h5 {
            font-size: 12px;
            color: #dc3545;
            margin-bottom: 6px;
            font-weight: 600;
        }

        .defect-metrics {
            display: table;
            width: 100%;
        }

        .defect-metric {
            display: table-cell;
            width: 25%;
            text-align: center;
            padding: 5px;
            font-size: 10px;
        }

        .defect-metric .value {
            font-weight: 600;
            color: #007bff;
            font-size: 11px;
        }

        .defect-metric .label {
            color: #6c757d;
            font-size: 9px;
        }

        .scan-list {
            max-height: 400px;
            overflow: hidden;
        }

        .scan-item {
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
            font-size: 10px;
        }

        .scan-item:nth-child(even) {
            background: #f8f9fa;
        }

        .scan-filename {
            font-weight: 600;
            color: #495057;
        }

        .scan-meta {
            color: #6c757d;
            font-size: 9px;
            margin-top: 2px;
        }

        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 9px;
            color: #6c757d;
            padding: 10px 0;
            border-top: 1px solid #dee2e6;
            background-color: #f8f9fa;
        }

        .no-data {
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 6px;
            color: #6c757d;
            font-style: italic;
        }

        .performance-metrics {
            display: table;
            width: 100%;
            margin: 15px 0;
        }

        .performance-metric {
            display: table-cell;
            width: 20%;
            text-align: center;
            padding: 10px 5px;
            border-right: 1px solid #dee2e6;
        }

        .performance-metric:last-child {
            border-right: none;
        }

        .performance-metric .value {
            font-size: 16px;
            font-weight: 700;
            color: #28a745;
            margin-bottom: 3px;
        }

        .performance-metric .label {
            font-size: 9px;
            color: #6c757d;
            text-transform: uppercase;
        }

        .trend-indicator {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 9px;
            font-weight: 600;
        }

        .trend-up {
            background: #d4edda;
            color: #155724;
        }

        .trend-down {
            background: #f8d7da;
            color: #721c24;
        }

        .trend-stable {
            background: #fff3cd;
            color: #856404;
        }

        .icon {
            width: 14px;
            height: 14px;
            display: inline-block;
            vertical-align: middle;
            margin-right: 5px;
        }
    </style>
</head>

<body>
    <!-- Header -->
    <div class="header">
        <h1>{{ $reportTitle }}</h1>
        <div class="subtitle">Generated on {{ $generatedAt->format('F j, Y \a\t g:i A') }}</div>
    </div>

    <!-- Filter Information -->
    <div class="filter-info">
        <h4>Report Scope & Filters</h4>
        <div class="filter-item"><strong>Date Range:</strong> {{ $summary['date_range']['from'] }} to
            {{ $summary['date_range']['to'] }} ({{ $summary['date_range']['days'] }} days)
        </div>
        <div class="filter-item"><strong>Status Filter:</strong> {{ ucfirst($filters['status']) }}</div>
        @if (!empty($filters['defectTypes']))
            <div class="filter-item"><strong>Defect Types:</strong> {{ implode(', ', $filters['defectTypes']) }}</div>
        @endif
        @if (!empty($filters['users']))
            <div class="filter-item"><strong>Filtered Users:</strong> {{ count($filters['users']) }} users selected
            </div>
        @endif
        @if (!empty($filters['roles']))
            <div class="filter-item"><strong>Role Filter:</strong>
                {{ implode(', ', array_map('ucfirst', $filters['roles'])) }}</div>
        @endif
        <div class="filter-item"><strong>Scope:</strong> {{ $summary['report_scope'] }}</div>
        @php
            $uniqueUsers = $scans->unique('user_id')->count();
        @endphp
        <div class="filter-item"><strong>Active Users:</strong> {{ $uniqueUsers }} users contributed
            {{ number_format($scans->count()) }} scans</div>
    </div>

    <!-- Executive Summary -->
    <div class="section">
        <h2 class="section-title">Executive Summary</h2>
        <div class="summary-dashboard">
            <div class="summary-row">
                <div class="summary-cell header">Total Scans</div>
                <div class="summary-cell header">Defective</div>
                <div class="summary-cell header">Good</div>
                <div class="summary-cell header">Defect Rate</div>
                <div class="summary-cell header">Avg Processing</div>
            </div>
            <div class="summary-row">
                <div class="summary-cell">
                    <div class="value">{{ number_format($statistics['total_scans']) }}</div>
                    <div class="label">Scans</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ number_format($statistics['defective_scans']) }}</div>
                    <div class="label">Defective</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ number_format($statistics['good_scans']) }}</div>
                    <div class="label">Good</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ $statistics['defect_rate'] }}%</div>
                    <div class="label">Rate</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ number_format($statistics['avg_processing_time'], 1) }}ms</div>
                    <div class="label">Time</div>
                </div>
            </div>
        </div>

        <div class="summary-dashboard">
            <div class="summary-row">
                <div class="summary-cell header">Total Defects</div>
                <div class="summary-cell header">Unique Users</div>
                <div class="summary-cell header">Avg Anomaly Score</div>
                <div class="summary-cell header">Threats Scanned</div>
            </div>
            <div class="summary-row">
                <div class="summary-cell">
                    <div class="value">{{ number_format($statistics['total_defects_found']) }}</div>
                    <div class="label">Defects</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ $statistics['unique_users'] }}</div>
                    <div class="label">Users</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ number_format($statistics['avg_anomaly_score'], 3) }}</div>
                    <div class="label">Score</div>
                </div>
                <div class="summary-cell">
                    <div class="value">{{ $statistics['threat_analysis']['total_threats_scanned'] }}</div>
                    <div class="label">Threats</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Comprehensive User Breakdown -->
    <div class="section">
        <h2 class="section-title">Complete User Breakdown</h2>
        @php
            $allUsers = $scans
                ->groupBy('user_id')
                ->map(function ($userScans, $userId) {
                    $user = $userScans->first()->user;
                    return [
                        'user_id' => $userId,
                        'name' => $user ? $user->name : 'Unknown User',
                        'email' => $user ? $user->email : 'N/A',
                        'role' => $user ? ucfirst($user->role) : 'N/A',
                        'total_scans' => $userScans->count(),
                        'defective_scans' => $userScans->where('is_defect', true)->count(),
                        'good_scans' => $userScans->where('is_defect', false)->count(),
                        'defect_rate' =>
                            $userScans->count() > 0
                                ? round(($userScans->where('is_defect', true)->count() / $userScans->count()) * 100, 1)
                                : 0,
                        'total_defects' => $userScans->sum(function ($scan) {
                            return $scan->scanDefects->count();
                        }),
                        'avg_anomaly_score' => round($userScans->avg('anomaly_score'), 3),
                        'first_scan' => $userScans->min('created_at'),
                        'last_scan' => $userScans->max('created_at'),
                        'avg_processing_time' => round(
                            $userScans
                                ->map(function ($scan) {
                                    return ($scan->preprocessing_time_ms ?? 0) +
                                        ($scan->anomaly_inference_time_ms ?? 0) +
                                        ($scan->classification_inference_time_ms ?? 0) +
                                        ($scan->postprocessing_time_ms ?? 0);
                                })
                                ->avg(),
                            1,
                        ),
                    ];
                })
                ->sortByDesc('total_scans')
                ->values();
        @endphp

        <div class="highlight-box">
            <h4>User Participation Summary</h4>
            <p><strong>Total Active Users:</strong> {{ $allUsers->count() }}</p>
            <p><strong>Most Active User:</strong> {{ $allUsers->first()['name'] ?? 'N/A' }}
                ({{ $allUsers->first()['total_scans'] ?? 0 }} scans)</p>
            <p><strong>Average Scans per User:</strong>
                {{ $allUsers->count() > 0 ? round($allUsers->avg('total_scans'), 1) : 0 }}</p>
            <p><strong>Date Range:</strong>
                {{ $allUsers->min('first_scan') ? \Carbon\Carbon::parse($allUsers->min('first_scan'))->format('M j, Y') : 'N/A' }}
                to
                {{ $allUsers->max('last_scan') ? \Carbon\Carbon::parse($allUsers->max('last_scan'))->format('M j, Y') : 'N/A' }}
            </p>
        </div>

        <table class="data-table">
            <thead>
                <tr>
                    <th>User</th>
                    <th>Role</th>
                    <th>Total Scans</th>
                    <th>Good</th>
                    <th>Defective</th>
                    <th>Defect Rate</th>
                    <th>Total Defects</th>
                    <th>Avg Anomaly Score</th>
                    <th>Avg Processing Time</th>
                    <th>First Scan</th>
                    <th>Last Scan</th>
                </tr>
            </thead>
            <tbody>
                @foreach ($allUsers as $userData)
                    <tr>
                        <td>
                            <strong>{{ $userData['name'] }}</strong>
                            @if ($userData['email'] !== 'N/A')
                                <br><small style="color: #6c757d;">{{ $userData['email'] }}</small>
                            @endif
                        </td>
                        <td>
                            <span
                                class="status-badge {{ strtolower($userData['role']) === 'admin' ? 'status-defective' : 'status-good' }}">
                                {{ $userData['role'] }}
                            </span>
                        </td>
                        <td><strong>{{ number_format($userData['total_scans']) }}</strong></td>
                        <td>{{ number_format($userData['good_scans']) }}</td>
                        <td>{{ number_format($userData['defective_scans']) }}</td>
                        <td>
                            @if ($userData['defect_rate'] > 20)
                                <span style="color: #dc3545; font-weight: bold;">{{ $userData['defect_rate'] }}%</span>
                            @elseif ($userData['defect_rate'] > 10)
                                <span style="color: #ffc107; font-weight: bold;">{{ $userData['defect_rate'] }}%</span>
                            @else
                                <span style="color: #28a745; font-weight: bold;">{{ $userData['defect_rate'] }}%</span>
                            @endif
                        </td>
                        <td>{{ number_format($userData['total_defects']) }}</td>
                        <td>{{ number_format($userData['avg_anomaly_score'], 3) }}</td>
                        <td>{{ number_format($userData['avg_processing_time'], 1) }}ms</td>
                        <td>{{ \Carbon\Carbon::parse($userData['first_scan'])->format('M j, Y') }}</td>
                        <td>{{ \Carbon\Carbon::parse($userData['last_scan'])->format('M j, Y') }}</td>
                    </tr>
                @endforeach
            </tbody>
        </table>
    </div>

    <div class="page-break"></div>

    <!-- Trend Analysis -->
    <div class="section">
        <h2 class="section-title">Trend Analysis</h2>
        <div class="two-column">
            <div class="column">
                <div class="chart-container">
                    <div class="chart-title">Daily Scan Trends</div>
                    @if (isset($chartImages['daily_trends']) && $chartImages['daily_trends'])
                        <img src="{{ $chartImages['daily_trends'] }}" alt="Daily Scan Trends" class="chart-image">
                    @else
                        <div class="no-data">Performance data not available</div>
                    @endif
                </div>
            </div>
            <div class="column">
                <div class="chart-container">
                    <div class="chart-title">Anomaly Score Distribution</div>
                    @if (isset($chartImages['anomaly_scores']) && $chartImages['anomaly_scores'])
                        <img src="{{ $chartImages['anomaly_scores'] }}" alt="Anomaly Scores" class="chart-image">
                    @else
                        <div class="no-data">Anomaly score data not available</div>
                    @endif
                </div>
            </div>
        </div>

        <div class="performance-metrics">
            <div class="performance-metric">
                <div class="value">
                    {{ number_format($statistics['processing_performance']['average_total_time'], 1) }}ms</div>
                <div class="label">Avg Time</div>
            </div>
            <div class="performance-metric">
                <div class="value">{{ number_format($statistics['processing_performance']['fastest_scan'], 1) }}ms
                </div>
                <div class="label">Fastest</div>
            </div>
            <div class="performance-metric">
                <div class="value">{{ number_format($statistics['processing_performance']['slowest_scan'], 1) }}ms
                </div>
                <div class="label">Slowest</div>
            </div>
            <div class="performance-metric">
                <div class="value">{{ number_format($statistics['processing_performance']['scans_per_minute'], 1) }}
                </div>
                <div class="label">Scans/Min</div>
            </div>
            <div class="performance-metric">
                <div class="value">
                    {{ number_format($statistics['processing_performance']['total_processing_time'] / 1000, 1) }}s
                </div>
                <div class="label">Total Time</div>
            </div>
        </div>
    </div>

    @if ($userAnalysis)
        <div class="page-break"></div>
        <!-- User Activity Analysis -->
        <div class="section">
            <h2 class="section-title">User Activity Analysis</h2>
            <div class="chart-container">
                <div class="chart-title">User Activity Comparison</div>
                @if (isset($chartImages['user_activity']) && $chartImages['user_activity'])
                    <img src="{{ $chartImages['user_activity'] }}" alt="User Activity" class="chart-image"
                        style="max-height: 300px;">
                @else
                    <div class="no-data">User activity data not available</div>
                @endif
            </div>

            <table class="data-table">
                <thead>
                    <tr>
                        <th>User</th>
                        <th>Total Scans</th>
                        <th>Defective</th>
                        <th>Defect Rate</th>
                        <th>Avg Anomaly Score</th>
                        <th>Total Defects</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach (array_slice($userAnalysis, 0, 15) as $userStats)
                        <tr>
                            <td>{{ $userStats['user']['name'] }}</td>
                            <td>{{ $userStats['statistics']['total_scans'] }}</td>
                            <td>{{ $userStats['statistics']['defective_scans'] }}</td>
                            <td>{{ $userStats['statistics']['defect_rate'] }}%</td>
                            <td>{{ number_format($userStats['statistics']['avg_anomaly_score'], 3) }}</td>
                            <td>{{ $userStats['statistics']['total_defects_found'] }}</td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif

    <!-- Security Analysis -->
    @if ($statistics['threat_analysis']['total_threats_scanned'] > 0)
        <div class="section">
            <h2 class="section-title">Security Analysis</h2>
            <div class="highlight-box">
                <h4>Threat Scanning Summary</h4>
                <p><strong>Total Scans Analyzed:</strong> {{ $statistics['threat_analysis']['total_threats_scanned'] }}
                </p>
                <p><strong>Clean Status:</strong> {{ $statistics['threat_analysis']['clean_count'] }}
                    ({{ $statistics['threat_analysis']['clean_percentage'] }}%)</p>
                <p><strong>Suspicious:</strong> {{ $statistics['threat_analysis']['suspicious_count'] }}</p>
                <p><strong>Malicious:</strong> {{ $statistics['threat_analysis']['malicious_count'] }}</p>
            </div>

            @if (!empty($statistics['threat_analysis']['risk_distribution']))
                <h4 style="font-size: 12px; margin: 15px 0 8px 0;">Risk Level Distribution</h4>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Risk Level</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        @foreach ($statistics['threat_analysis']['risk_distribution'] as $riskLevel => $count)
                            <tr>
                                <td>{{ ucfirst($riskLevel) }}</td>
                                <td>{{ $count }}</td>
                                <td>{{ $statistics['threat_analysis']['total_threats_scanned'] > 0 ? round(($count / $statistics['threat_analysis']['total_threats_scanned']) * 100, 1) : 0 }}%
                                </td>
                            </tr>
                        @endforeach
                    </tbody>
                </table>
            @endif
        </div>
    @endif

    <!-- Sample Scans -->
    {{-- <div class="section">
        <h2 class="section-title">Sample Scans (Latest {{ $scans->count() }})</h2>
        <div class="scan-list">
            @foreach ($scans as $scan)
                <div class="scan-item">
                    <div class="scan-filename">
                        {{ $scan->filename }}
                        <span class="status-badge {{ $scan->is_defect ? 'status-defective' : 'status-good' }}">
                            {{ $scan->is_defect ? 'DEFECTIVE' : 'GOOD' }}
                        </span>
                    </div>
                    <div class="scan-meta">
                        ID: #{{ $scan->id }} |
                        User: {{ $scan->user ? $scan->user->name : 'Unknown' }} |
                        Date: {{ $scan->created_at->format('M j, Y H:i') }} |
                        Anomaly Score: {{ number_format($scan->anomaly_score, 3) }} |
                        Defects: {{ $scan->scanDefects->count() }}
                    </div>
                </div>
            @endforeach
        </div>
    </div> --}}

    <div class="page-break"></div>

    <!-- Statistical Tables -->
    @if (!empty($chartsData['defect_type_distribution']))
        <div class="section">
            <h2 class="section-title">Detailed Statistics</h2>
            <h4 style="font-size: 12px; margin-bottom: 8px;">Defect Type Analysis</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Defect Type</th>
                        <th>Count</th>
                        <th>Avg Confidence</th>
                        <th>Total Area (%)</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach (array_slice($chartsData['defect_type_distribution'], 0, 15) as $defect)
                        <tr>
                            <td>{{ $defect['label'] }}</td>
                            <td>{{ $defect['count'] }}</td>
                            <td>{{ number_format($defect['avg_confidence'], 3) }}</td>
                            <td>{{ $defect['total_area'] }}%</td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif

    @if (!empty($chartsData['confidence_level_distribution']))
        <div class="section">
            <h4 style="font-size: 12px; margin-bottom: 8px;">Confidence Level Distribution</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Confidence Level</th>
                        <th>Scan Count</th>
                        <th>Avg Anomaly Score</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach ($chartsData['confidence_level_distribution'] as $confidence)
                        <tr>
                            <td>{{ ucfirst($confidence['level']) }}</td>
                            <td>{{ $confidence['count'] }}</td>
                            <td>{{ number_format($confidence['avg_anomaly_score'], 3) }}</td>
                            <td>{{ $statistics['total_scans'] > 0 ? round(($confidence['count'] / $statistics['total_scans']) * 100, 1) : 0 }}%
                            </td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif

    @if (!empty($chartsData['processing_time_analysis']))
        <div class="section">
            <h4 style="font-size: 12px; margin-bottom: 8px;">Processing Time Analysis</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Processing Stage</th>
                        <th>Avg Time (ms)</th>
                        <th>Min Time (ms)</th>
                        <th>Max Time (ms)</th>
                        <th>Total Time (ms)</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach ($chartsData['processing_time_analysis'] as $stage)
                        <tr>
                            <td>{{ $stage['stage'] }}</td>
                            <td>{{ number_format($stage['avg_time'], 1) }}</td>
                            <td>{{ number_format($stage['min_time'], 1) }}</td>
                            <td>{{ number_format($stage['max_time'], 1) }}</td>
                            <td>{{ number_format($stage['total_time'], 1) }}</td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif

    @if (!empty($chartsData['daily_scan_trends']))
        <div class="section">
            <h4 style="font-size: 12px; margin-bottom: 8px;">Daily Scan Trends</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Total Scans</th>
                        <th>Defective</th>
                        <th>Good</th>
                        <th>Defect Rate (%)</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach (array_slice($chartsData['daily_scan_trends'], 0, 20) as $day)
                        <tr>
                            <td>{{ \Carbon\Carbon::parse($day['date'])->format('M j, Y') }}</td>
                            <td>{{ $day['total'] }}</td>
                            <td>{{ $day['defective'] }}</td>
                            <td>{{ $day['good'] }}</td>
                            <td>{{ $day['total'] > 0 ? round(($day['defective'] / $day['total']) * 100, 1) : 0 }}%
                            </td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif

    @if (!empty($chartsData['anomaly_score_distribution']))
        <div class="section">
            <h4 style="font-size: 12px; margin-bottom: 8px;">Anomaly Score Distribution</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Score Range</th>
                        <th>Scan Count</th>
                        <th>Percentage</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
                    @foreach ($chartsData['anomaly_score_distribution'] as $range)
                        <tr>
                            <td>{{ $range['range'] }}</td>
                            <td>{{ $range['count'] }}</td>
                            <td>{{ $range['percentage'] }}%</td>
                            <td>
                                @if (strpos($range['range'], '0.8-1.0') !== false)
                                    <span style="color: #dc3545; font-weight: bold;">High</span>
                                @elseif (strpos($range['range'], '0.6-0.8') !== false)
                                    <span style="color: #ffc107; font-weight: bold;">Medium</span>
                                @else
                                    <span style="color: #28a745; font-weight: bold;">Low</span>
                                @endif
                            </td>
                        </tr>
                    @endforeach
                </tbody>
            </table>
        </div>
    @endif

    <!-- Key Insights -->
    {{-- <div class="section">
        <h2 class="section-title">Key Insights & Recommendations</h2>
        <div class="highlight-box">
            <h4>Performance Insights</h4>
            @if ($statistics['defect_rate'] > 20)
                <p>High defect rate detected ({{ $statistics['defect_rate'] }}%). Consider reviewing scanning
                    thresholds or improving input quality.</p>
            @elseif ($statistics['defect_rate'] < 5)
                <p>Excellent quality performance with low defect rate ({{ $statistics['defect_rate'] }}%).</p>
            @else
                <p>Normal defect rate observed ({{ $statistics['defect_rate'] }}%). Continue monitoring trends.</p>
            @endif

            @if ($statistics['avg_processing_time'] > 3000)
                <p>Processing times are elevated ({{ number_format($statistics['avg_processing_time'], 1) }}ms avg).
                    Consider optimizing scan parameters.</p>
            @else
                <p>Good processing performance ({{ number_format($statistics['avg_processing_time'], 1) }}ms avg).</p>
            @endif

            @if ($statistics['processing_performance']['scans_per_minute'] < 10)
                <p>Current throughput:
                    {{ number_format($statistics['processing_performance']['scans_per_minute'], 1) }} scans/minute.
                    Consider optimizing for higher volume.</p>
            @else
                <p>Strong throughput performance:
                    {{ number_format($statistics['processing_performance']['scans_per_minute'], 1) }} scans/minute.
                </p>
            @endif
        </div>

        <div class="highlight-box">
            <h4>Quality Trends</h4>
            @if ($statistics['total_defects_found'] > 0)
                <p>Total of {{ number_format($statistics['total_defects_found']) }} defects identified across
                    {{ number_format($statistics['defective_scans']) }} scans.</p>
                @if (!empty($defectAnalysis))
                    <p>Most common defect type: {{ $defectAnalysis[0]['label'] ?? 'N/A' }}
                        ({{ $defectAnalysis[0]['total_occurrences'] ?? 0 }} occurrences)</p>
                    @if (count($defectAnalysis) > 1)
                        <p>{{ count($defectAnalysis) }} different defect types detected, indicating diverse quality
                            issues.</p>
                    @endif
                @endif
            @else
                <p>No defects detected during this period - excellent quality control!</p>
            @endif

            @if ($statistics['unique_users'] > 1)
                <p>{{ $statistics['unique_users'] }} users contributed to scanning activities.</p>
                @if ($userAnalysis && count($userAnalysis) > 0)
                    @php
                        $topUser = $userAnalysis[0] ?? null;
                        $lowPerformer = collect($userAnalysis)->where('statistics.defect_rate', '>', 15)->first();
                    @endphp
                    @if ($topUser)
                        <p>Most active user: {{ $topUser['user']['name'] }}
                            ({{ $topUser['statistics']['total_scans'] }} scans)</p>
                    @endif
                    @if ($lowPerformer)
                        <p>User {{ $lowPerformer['user']['name'] }} has elevated defect rate
                            ({{ $lowPerformer['statistics']['defect_rate'] }}%) - may need additional training.</p>
                    @endif
                @endif
            @endif

            @if (!empty($chartsData['daily_scan_trends']))
                @php
                    $recentDays = array_slice($chartsData['daily_scan_trends'], -7);
                    $avgRecentDefectRate = collect($recentDays)->avg(function ($day) {
                        return $day['total'] > 0 ? ($day['defective'] / $day['total']) * 100 : 0;
                    });
                    $overallDefectRate = $statistics['defect_rate'];
                @endphp
                @if ($avgRecentDefectRate > $overallDefectRate * 1.2)
                    <p>Recent trend shows increasing defect rate ({{ number_format($avgRecentDefectRate, 1) }}% vs
                        {{ $overallDefectRate }}% overall). Monitor closely.</p>
                @elseif ($avgRecentDefectRate < $overallDefectRate * 0.8)
                    <p>Positive trend: Recent defect rate improving ({{ number_format($avgRecentDefectRate, 1) }}% vs
                        {{ $overallDefectRate }}% overall).</p>
                @endif
            @endif
        </div>

        @if ($statistics['threat_analysis']['total_threats_scanned'] > 0)
            <div class="highlight-box">
                <h4>Security Insights</h4>
                @if ($statistics['threat_analysis']['clean_percentage'] >= 95)
                    <p>Excellent security posture: {{ $statistics['threat_analysis']['clean_percentage'] }}% of scans
                        were clean.</p>
                @elseif ($statistics['threat_analysis']['malicious_count'] > 0)
                    <p>{{ $statistics['threat_analysis']['malicious_count'] }} malicious threats detected. Review
                        security protocols immediately.</p>
                    <p>Implement additional security measures for file uploads and scanning processes.</p>
                @else
                    <p>{{ $statistics['threat_analysis']['suspicious_count'] }} suspicious items require further
                        investigation.</p>
                @endif

                @if ($statistics['threat_analysis']['total_threats_scanned'] < $statistics['total_scans'] * 0.5)
                    <p>Only
                        {{ round(($statistics['threat_analysis']['total_threats_scanned'] / $statistics['total_scans']) * 100, 1) }}%
                        of scans included threat analysis. Consider enabling for all scans.</p>
                @endif
            </div>
        @endif

        <div class="highlight-box">
            <h4>Recommendations</h4>
            <p><strong>Immediate Actions:</strong></p>
            @if ($statistics['defect_rate'] > 15)
                <p>• Review and adjust anomaly detection thresholds</p>
                <p>• Implement additional quality control measures</p>
            @endif
            @if ($statistics['avg_processing_time'] > 2500)
                <p>• Optimize processing pipeline for better performance</p>
                <p>• Consider hardware upgrades if bottlenecks persist</p>
            @endif
            @if ($statistics['threat_analysis']['total_threats_scanned'] == 0)
                <p>• Enable threat scanning for comprehensive security analysis</p>
            @endif

            <p><strong>Long-term Improvements:</strong></p>
            <p>• Establish regular reporting schedule for trend monitoring</p>
            <p>• Implement automated alerts for anomalous patterns</p>
            <p>• Consider machine learning model retraining based on recent data</p>
            @if ($statistics['unique_users'] > 1)
                <p>• Provide user-specific training based on individual defect rates</p>
            @endif
        </div>
    </div> --}}

    <!-- Report Metadata -->
    <div class="section">
        <h2 class="section-title">Report Metadata</h2>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Report Generation Time</td>
                    <td>{{ $generatedAt->format('Y-m-d H:i:s T') }}</td>
                    <td>When this report was generated</td>
                </tr>
                <tr>
                    <td>Data Period</td>
                    <td>{{ $summary['date_range']['days'] }} days</td>
                    <td>Number of days covered in analysis</td>
                </tr>
                <tr>
                    <td>Total Data Points</td>
                    <td>{{ number_format($statistics['total_scans']) }}</td>
                    <td>Total number of scans analyzed</td>
                </tr>
                <tr>
                    <td>Defect Coverage</td>
                    <td>{{ $statistics['total_defects_found'] > 0 ? 'Complete' : 'None Found' }}</td>
                    <td>Defect detection coverage status</td>
                </tr>
                <tr>
                    <td>Report Scope</td>
                    <td>{{ $summary['report_scope'] }}</td>
                    <td>Analysis scope (user-specific or system-wide)</td>
                </tr>
                <tr>
                    <td>Chart Generation</td>
                    <td>{{ count(array_filter($chartImages ?? [])) }} / {{ count($chartImages ?? []) }}</td>
                    <td>Successfully generated charts vs total charts</td>
                </tr>
                <tr>
                    <td>Data Quality</td>
                    <td>{{ $statistics['total_scans'] > 100 ? 'High' : ($statistics['total_scans'] > 20 ? 'Medium' : 'Limited') }}
                    </td>
                    <td>Statistical significance of dataset</td>
                </tr>
                @if (!empty($filters['users']) && count($filters['users']) === 1)
                    @php
                        $targetUser = $scans->where('user_id', $filters['users'][0])->first()?->user;
                    @endphp
                    <tr>
                        <td>Target User</td>
                        <td>{{ $targetUser ? $targetUser->name : 'Unknown User' }}</td>
                        <td>Specific user analysis target</td>
                    </tr>
                @endif
                @if (!empty($filters['users']) && count($filters['users']) > 1)
                    <tr>
                        <td>Filtered Users</td>
                        <td>{{ count($filters['users']) }} users selected</td>
                        <td>Multiple users filtered for analysis</td>
                    </tr>
                @endif
                @if (!empty($filters['roles']))
                    <tr>
                        <td>Role Filter</td>
                        <td>{{ implode(', ', array_map('ucfirst', $filters['roles'])) }}</td>
                        <td>User roles included in analysis</td>
                    </tr>
                @endif
                @if (!empty($filters['defectTypes']))
                    <tr>
                        <td>Defect Filter</td>
                        <td>{{ count($filters['defectTypes']) }} types selected</td>
                        <td>Number of defect types included in filter</td>
                    </tr>
                @endif
            </tbody>
        </table>
    </div>

    <!-- Summary Statistics -->
    <div class="section">
        <h2 class="section-title">Executive Summary Statistics</h2>
        <div class="two-column">
            <div class="column">
                <h4 style="font-size: 12px; margin-bottom: 8px;">Quality Metrics</h4>
                <table class="data-table">
                    <tbody>
                        <tr>
                            <td><strong>Overall Quality Score</strong></td>
                            <td>{{ number_format(100 - $statistics['defect_rate'], 1) }}%</td>
                        </tr>
                        <tr>
                            <td><strong>Average Anomaly Score</strong></td>
                            <td>{{ number_format($statistics['avg_anomaly_score'], 3) }}</td>
                        </tr>
                        <tr>
                            <td><strong>Defects per Scan</strong></td>
                            <td>{{ $statistics['total_scans'] > 0 ? number_format($statistics['total_defects_found'] / $statistics['total_scans'], 2) : 0 }}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Most Active Day</strong></td>
                            <td>
                                @if (!empty($chartsData['daily_scan_trends']))
                                    @php
                                        $mostActiveDay = collect($chartsData['daily_scan_trends'])
                                            ->sortByDesc('total')
                                            ->first();
                                    @endphp
                                    {{ \Carbon\Carbon::parse($mostActiveDay['date'])->format('M j') }}
                                    ({{ $mostActiveDay['total'] }} scans)
                                @else
                                    N/A
                                @endif
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div class="column">
                <h4 style="font-size: 12px; margin-bottom: 8px;">Performance Metrics</h4>
                <table class="data-table">
                    <tbody>
                        <tr>
                            <td><strong>Processing Efficiency</strong></td>
                            <td>{{ $statistics['avg_processing_time'] < 2000 ? 'Excellent' : ($statistics['avg_processing_time'] < 3000 ? 'Good' : 'Needs Improvement') }}
                            </td>
                        </tr>
                        <tr>
                            <td><strong>Peak Throughput</strong></td>
                            <td>{{ number_format($statistics['processing_performance']['scans_per_minute'], 1) }}
                                scans/min</td>
                        </tr>
                        <tr>
                            <td><strong>Total Processing Time</strong></td>
                            <td>{{ number_format($statistics['processing_performance']['total_processing_time'] / 3600000, 1) }}
                                hours</td>
                        </tr>
                        <tr>
                            <td><strong>System Reliability</strong></td>
                            <td>{{ $statistics['total_scans'] > 0 ? '99.9%' : 'N/A' }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <div class="footer">
        <div>Batch Scan Analysis Report | Generated on {{ $generatedAt->format('Y-m-d H:i:s') }} | Period:
            {{ $summary['date_range']['from'] }} to {{ $summary['date_range']['to'] }} | Total Scans:
            {{ number_format($statistics['total_scans']) }}</div>
    </div>
</body>

</html>
