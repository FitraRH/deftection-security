<?php

namespace App\Http\Controllers;

use Inertia\Inertia;
use Illuminate\Http\Request;

class RealtimeScanController extends Controller
{
    //
    public function index()
    {
        // This method can be used to list all realtime scans if needed
        return Inertia::render('DetailSession/Index');
    }
}
