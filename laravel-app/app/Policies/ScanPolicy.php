<?php

namespace App\Policies;

use App\Models\Scan;
use App\Models\User;
use Illuminate\Auth\Access\HandlesAuthorization;

class ScanPolicy
{
    use HandlesAuthorization;

    /**
     * Determine whether the user can view any scans.
     */
    public function viewAny(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can view the scan.
     */
    public function view(User $user, Scan $scan): bool
    {
        return $user->role === 'admin' || $scan->user_id === $user->id;
    }

    /**
     * Determine whether the user can delete the scan.
     */
    public function delete(User $user, Scan $scan): bool
    {
        return $user->role === 'admin' || $scan->user_id === $user->id;
    }

    /**
     * Determine whether the user can delete any scans.
     */
    public function deleteAny(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can filter by user (admin only).
     */
    public function filterByUser(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can generate a report for a specific scan.
     */
    public function generateReport(User $user, Scan $scan): bool
    {
        // User can generate report for their own scans, admin can generate for any scan
        return $user->role === 'admin' || $scan->user_id === $user->id;
    }

    /**
     * Determine whether the user can generate batch reports for any user.
     */
    public function generateAnyReport(User $user): bool
    {
        return $user->role === 'admin';
    }

    /**
     * Determine whether the user can generate batch reports.
     * Users can generate batch reports for their own data, admins for any data.
     */
    public function generateBatchReport(User $user): bool
    {
        return true;
    }
}
