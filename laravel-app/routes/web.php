<?php

use Inertia\Inertia;
use App\Models\DefectType;
use Illuminate\Support\Facades\Route;
use App\Http\Middleware\RoleMiddleware;
use App\Http\Controllers\ScanController;
use App\Http\Controllers\UserController;
use App\Http\Controllers\ReportController;
use App\Http\Controllers\ProductController;
use App\Http\Controllers\RealtimeController;
use App\Http\Controllers\SettingsController;
use App\Http\Controllers\Auth\LoginController;
use App\Http\Controllers\DefectTypeController;
use App\Http\Controllers\RealtimeScanController;
use App\Http\Controllers\Auth\RegisterController;
use App\Http\Controllers\RealtimeAnalysisController;

Route::permanentRedirect('/', '/login');
Route::post('/logout', [LoginController::class, 'destroy'])->middleware('auth')->name('logout');

Route::middleware(['guest'])->group(function () {
  Route::get('/login', [LoginController::class, 'create'])->name('login');
  Route::post('/login', [LoginController::class, 'store'])->name('login.store');
  Route::get('/register', [RegisterController::class, 'create'])->name('register');
  Route::post('/register', [RegisterController::class, 'store'])->name('register.store');
});

Route::middleware(['auth'])->group(function () {
  //DASHBOARD
  Route::get('/dashboard', function () {
    return Inertia::render('Dashboard/Index');
  })->name('dashboard');

  // PRODUCTS
  Route::prefix('database/products')->group(function () {
    // Product list
    Route::get('/', [ProductController::class, 'index'])->name('products.index');
    Route::get('/api', [ProductController::class, 'indexApi'])->middleware('throttle:60,1')->name('products.index.api');
    Route::get('/check-updates', [ProductController::class, 'indexCheck'])->middleware('throttle:120,1')->name('products.index.check');
    Route::get('/force-refresh', [ProductController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('products.index.refresh');
    Route::get('/{product}', [ProductController::class, 'showApi'])->middleware('throttle:60,1')->name('products.show.api');
    // Product operations
    Route::post('/', [ProductController::class, 'store'])->name('products.store');
    Route::put('/{product}', [ProductController::class, 'update'])->name('products.update');
    Route::delete('/{product}', [ProductController::class, 'destroy'])->name('products.destroy');
  });

  // DEFECT TYPES
  Route::prefix('database/defectTypes')->group(function () {
    // Defect Type list
    Route::get('/', [DefectTypeController::class, 'index'])->name('defect_types.index');
    Route::get('/api', [DefectTypeController::class, 'indexApi'])->middleware('throttle:60,1')->name('defect_types.index.api');
    Route::get('/check-updates', [DefectTypeController::class, 'indexCheck'])->middleware('throttle:120,1')->name('defect_types.index.check');
    Route::get('/force-refresh', [DefectTypeController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('defect_types.index.refresh');
    Route::get('/{defectType}', [DefectTypeController::class, 'showApi'])->middleware('throttle:60,1')->name('defect_types.show.api');
    // Defect Type operations
    Route::post('/', [DefectTypeController::class, 'store'])->name('defect_types.store');
    Route::put('/{defectType}', [DefectTypeController::class, 'update'])->name('defect_types.update');
    Route::delete('/{defectType}', [DefectTypeController::class, 'destroy'])->name('defect_types.destroy');
  });

  // USERS
  Route::prefix('database/users')->group(function () {
    // User list
    Route::get('/', [UserController::class, 'index'])->name('users.index');
    Route::get('/api', [UserController::class, 'indexApi'])->middleware('throttle:60,1')->name('users.index.api');
    Route::get('/check-updates', [UserController::class, 'indexCheck'])->middleware('throttle:120,1')->name('users.index.check');
    Route::get('/force-refresh', [UserController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('users.index.refresh');
    Route::get('/{user}', [UserController::class, 'showApi'])->middleware('throttle:60,1')->name('users.show.api');
    // User operations
    Route::post('/', [UserController::class, 'store'])->name('users.store');
    Route::put('/{user}', [UserController::class, 'update'])->name('users.update');
    Route::delete('/{user}', [UserController::class, 'destroy'])->name('users.destroy');
  });

  // DEFECT TYPES
  Route::prefix('database/defect-types')->group(function () {
    // User list
    Route::get('/', function () {
      return Inertia::render('Database/DefectTypes/Index', [
        'defectTypes' => [],
        'filters' => [],
        'meta' => [],
      ]);
    })->name('defect-types.index');
  });

  // REPORTS
  Route::prefix('reports')->name('reports.')->group(function () {

    // Single scan report
    Route::get('single/{scan}', [ReportController::class, 'generateSingleReport'])
      ->name('single.generate')
      ->middleware('can:generateReport,scan');

    // Batch report - requires authentication, authorization handled in controller
    Route::get('batch', [ReportController::class, 'generateBatchReport'])
      ->name('batch.generate')
      ->middleware('can:generateBatchReport,App\Models\Scan');

    // // Preview routes ( for test)
    // Route::get('preview/single/{scan}', [ReportController::class, 'previewSingleReport'])
    //   ->name('single.preview')
    //   ->middleware('can:generateReport,scan');

    // Route::get('preview/batch', [ReportController::class, 'previewBatchReport'])
    //   ->name('batch.preview')
    //   ->middleware('can:generateBatchReport,App\Models\Scan');
  });

  //SETTINGS
  Route::prefix('settings')->name('settings.')->group(function () {
    Route::get('/', [SettingsController::class, 'index'])->name('index');

    // Account Settings
    Route::patch('/account-name', [UserController::class, 'updateAccountName'])->name('account_name.update');
    Route::patch('/account-password', [UserController::class, 'updateAccountPassword'])->name('account_password.update');

    // Detection Settings
    Route::patch('/detection-settings', [SettingsController::class, 'update'])->name('detection_settings.update');

    // Clear Data
    Route::delete('/clear-data', [SettingsController::class, 'clearData'])->name('clear_all_data');
    Route::delete('/clear-my-data', [SettingsController::class, 'clearMyData'])->name('clear_my_data');
  });




  Route::get('/image-analysis', function () {
    return Inertia::render('ImageAnalysis/Index');
  })->name('image-analysis');


  // SCANS
  // scans store (image analysis)
  Route::get('/image-analysis', [ScanController::class, 'create'])->name('scans.create');
  Route::post('image-analysis', [ScanController::class, 'store'])->name('scans.store');
  // Scan list
  Route::prefix('analysis/scan-history')->group(function () {
    Route::get('/', [ScanController::class, 'index'])->name('scans.index');
    Route::get('/api', [ScanController::class, 'indexApi'])->middleware('throttle:60,1')->name('scans.index.api');
    Route::get('/check-updates', [ScanController::class, 'indexCheck'])->middleware('throttle:120,1')->name('scans.index.check');
    Route::get('/force-refresh', [ScanController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('scans.index.refresh');

    // Scan Details
    Route::get('/{scan}', [ScanController::class, 'show'])->name('scans.show');

    // Scan  operations
    Route::delete('/{scan}', [ScanController::class, 'destroy'])->name('scans.destroy');
  });


  // REALTIME SESSIONS  
  // Realtime session store (realtime analysis)
  Route::get('real-time-analysis', [RealtimeController::class, 'create'])->name('sessions.create');
  Route::post('real-time-analysis', [RealtimeController::class, 'store'])->name('sessions.store');
  // Session list
  Route::prefix('analysis/session-history')->group(function () {
    Route::get('/', [RealtimeController::class, 'index'])->name('sessions.index');
    Route::get('/api', [RealtimeController::class, 'indexApi'])->middleware('throttle:60,1')->name('sessions.index.api');
    Route::get('/check-updates', [RealtimeController::class, 'indexCheck'])->middleware('throttle:120,1')->name('sessions.index.check');
    Route::get('/force-refresh', [RealtimeController::class, 'indexRefresh'])->middleware('throttle:60,1')->name('sessions.index.refresh');

    // Realtime Session Details (Realtime scan index)
    Route::get('/{realtimeSession}', [RealtimeScanController::class, 'index'])->name('sessions_scan.index');
    // Realtime scan details
    Route::get('/{realtimeSession}/scan/{scan}', [RealtimeScanController::class, 'scan'])->name('sessions_scan.show');

    // Realtime Session  operations
    Route::delete('/{realtimeSession}', [RealtimeController::class, 'destroy'])->name('sessions.destroy');
  });


  // REALTIME SESSION API ENDPOINTS
  Route::prefix('api/realtime/sessions')->group(function () {
    // Start a new session
    Route::post('/start', [RealtimeAnalysisController::class, 'startSession'])
      ->name('realtime.sessions.start')
      ->middleware('throttle:10,1'); // Limit to 10 requests per minute

    // Pause a session
    Route::post('/pause', [RealtimeAnalysisController::class, 'pauseSession'])
      ->name('realtime.sessions.pause')
      ->middleware('throttle:10,1');

    Route::post('/{session}/pause', [RealtimeAnalysisController::class, 'pauseSession'])
      ->name('realtime.sessions.pause.specific')
      ->middleware('throttle:10,1');

    // Resume a session
    Route::post('/resume', [RealtimeAnalysisController::class, 'resumeSession'])
      ->name('realtime.sessions.resume')
      ->middleware('throttle:10,1');

    Route::post('/{session}/resume', [RealtimeAnalysisController::class, 'resumeSession'])
      ->name('realtime.sessions.resume.specific')
      ->middleware('throttle:10,1');

    // Stop a session (with optional session ID)
    Route::post('/stop', [RealtimeAnalysisController::class, 'stopSession'])
      ->name('realtime.sessions.stop')
      ->middleware('throttle:10,1');

    Route::post('/{session}/stop', [RealtimeAnalysisController::class, 'stopSession'])
      ->name('realtime.sessions.stop.specific')
      ->middleware('throttle:10,1');

    // Get current active session
    Route::get('/current', [RealtimeAnalysisController::class, 'getCurrentSession'])
      ->name('realtime.sessions.current')
      ->middleware('throttle:60,1'); // Allow more frequent checks

    // Get session status (for polling)
    Route::get('/{session}/status', [RealtimeAnalysisController::class, 'getSessionStatus'])
      ->name('realtime.sessions.status')
      ->middleware('throttle:120,1');
  });
});

Route::get('/home', function () {
  return Inertia::render('Database/ProductOld');
});
