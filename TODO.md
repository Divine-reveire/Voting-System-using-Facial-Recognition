# TODO: Add Reset Feature for Admin

## Steps to Complete

1. **Add /reset_data route in web_app.py**
   - Create a POST route that clears all stored data:
     - Delete Votes.csv
     - Clear data/candidates.pkl (empty list)
     - Clear data/voters.pkl (empty list)
     - Clear data/names.pkl (empty list)
     - Clear data/faces_data.pkl (empty array)
     - Clear data/voterids.pkl (empty list)
     - Remove all files in static/icons/ directory
     - Reload face data (set knn to None)
     - Log the reset action
   - Return success/error JSON response

2. **Update admin_manage.html**
   - Add a "Reset All Data" button in the admin management page
   - Add JavaScript for confirmation dialog before sending POST request to /reset_data
   - Handle response: reload page on success, show alert on error

3. **Test the reset functionality**
   - Run the app and test the reset button
   - Verify that all data is cleared and the app still functions normally
   - Check that face recognition is reset (knn = None)

4. **Final verification**
   - Ensure no data remains after reset
   - Confirm that adding new candidates/voters works after reset
