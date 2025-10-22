# TODO: Prevent Duplicate Face Registrations and Ensure Voting Prevention

## Steps to Complete

- [x] Modify `enter_details` in `web_app.py` to check Voter ID uniqueness by loading and checking `data/voterids.pkl`.
- [x] Update `save_face` in `web_app.py` to include Voter ID in the storage logic and add face similarity check using KNN distance threshold.
- [x] Create and initialize `data/voterids.pkl` if it doesn't exist, storing Voter IDs parallel to names.
- [x] Test registration flow to ensure duplicate Voter IDs and similar faces are rejected.
- [x] Test voting flow to confirm prevention works based on face recognition.
