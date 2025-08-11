<?php

namespace App\Http\Controllers;

use App\Models\Scan;
use App\Models\RealtimeSession;
use Illuminate\Support\Facades\DB;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Log;
use Inertia\Inertia;

class SettingsController extends Controller
{
    //
    public function index()
    {
        $defaults = [
            'detection' => [
                'anomalyThreshold' => 0.75,
                'defectThreshold' => 0.85,
                'exportFormat' => 'pdf',
            ]
        ];

        // Load settings from the JSON file, or use defaults if it's not found
        $settings = Storage::exists('settings.json')
            ? json_decode(Storage::get('settings.json'), true)
            : $defaults;

        // Render the Vue component and pass the loaded settings as a prop
        return Inertia::render('Settings/Index', [
            'savedSettings' => $settings
        ]);
    }

    public function update(Request $request)
    {
        // 1. Validate the incoming data to ensure it's in the correct format
        $validatedData = $request->validate([
            'detection.anomalyThreshold' => ['required', 'numeric', 'min:0', 'max:1'],
            'detection.defectThreshold' => ['required', 'numeric', 'min:0', 'max:1'],
            'detection.exportFormat' => ['required', 'string', 'in:pdf,json,csv'],
        ]);

        Log::info('Settings updated by user ID: ' . auth()->id(), $validatedData);

        // 2. Store the validated settings as a JSON file
        // This makes it easy for other services to read.
        Storage::disk('local')->put('settings.json', json_encode($validatedData, JSON_PRETTY_PRINT));

        // 3. Redirect back with a success message
        return back()->with('success', 'Settings saved successfully.');
    }

    public function clearData()
    {

        $this->authorize('deleteAny', RealtimeSession::class);
        $this->authorize('deleteAny', Scan::class);

        RealtimeSession::query()->delete();
        Scan::query()->delete();

        // Redirect back with a success message.
        return back()->with('success', 'All analysis data has been cleared.');
    }

    public function clearMyData()
    {
        $userId = auth()->id();

        $this->authorize('delete', new RealtimeSession(['user_id' => $userId]));
        RealtimeSession::where('user_id', $userId)->delete();

        $this->authorize('delete', new Scan(['user_id' => $userId]));
        Scan::where('user_id', $userId)->delete();



        // Redirect back with a success message.
        return back()->with('success', 'Your analysis data has been cleared.');
    }
}
