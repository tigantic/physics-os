"""
HERMENEUTIC UI VERIFICATION - DOCUMENT INGESTION E2E TEST
==========================================================

This test verifies THE CRITICAL FLOW that was broken:
1. User uploads a document containing HVAC specs
2. System extracts values from document  
3. User navigates to Configure step
4. Form fields are PRE-POPULATED with extracted values

This is NOT a unit test. This is a real browser automation that
clicks buttons and verifies visible UI state.

Evidence captured:
- Screenshots at each step
- Trace file for debugging
- Console logs
- Network requests
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, expect
import json
import tempfile

# Test configuration
BASE_URL = "http://localhost:8502"
EVIDENCE_DIR = Path(__file__).parent / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

# Test document content - this MUST appear in form fields after extraction
TEST_DOCUMENT_CONTENT = """
HVAC DESIGN SPECIFICATION
Project: Playwright Test Room
Client: E2E Verification Corp

ROOM DIMENSIONS:
- Length: 45 feet
- Width: 32 feet
- Height: 11 feet

HVAC REQUIREMENTS:
- Supply Airflow: 1800 CFM
- Supply Temperature: 58°F
- Number of Diffusers: 3

TARGET CONDITIONS:
- Temperature: 74°F
"""

# Expected extracted values - these MUST appear in form fields
EXPECTED_VALUES = {
    "room_length": 45.0,
    "room_width": 32.0,
    "room_height": 11.0,
    "supply_airflow": 1800.0,
}


async def test_document_ingestion_e2e():
    """
    THE CRITICAL TEST: Does document upload → form population actually work?
    """
    results = {
        "test": "document_ingestion_e2e",
        "steps": [],
        "passed": False,
        "failures": [],
    }
    
    async with async_playwright() as p:
        # Launch browser with tracing
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(EVIDENCE_DIR),
        )
        
        # Start tracing
        await context.tracing.start(screenshots=True, snapshots=True)
        
        page = await context.new_page()
        
        # Capture console logs
        console_logs = []
        page.on("console", lambda msg: console_logs.append(f"{msg.type}: {msg.text}"))
        
        try:
            # ============================================================
            # STEP 1: Navigate to app
            # ============================================================
            step1 = {"step": 1, "action": "Navigate to app", "status": "pending"}
            await page.goto(BASE_URL, wait_until="networkidle")
            await page.screenshot(path=str(EVIDENCE_DIR / "01_initial_load.png"))
            
            # Verify app loaded - look for the header
            header = page.locator("text=HyperFOAM")
            await expect(header.first).to_be_visible(timeout=10000)
            step1["status"] = "pass"
            step1["evidence"] = "01_initial_load.png"
            results["steps"].append(step1)
            print("✓ Step 1: App loaded")
            
            # ============================================================
            # STEP 2: Verify we're on Step 1 (Upload)
            # ============================================================
            step2 = {"step": 2, "action": "Verify on Upload step", "status": "pending"}
            
            # Look for file uploader
            uploader = page.locator('input[type="file"]')
            await expect(uploader).to_be_attached(timeout=5000)
            step2["status"] = "pass"
            results["steps"].append(step2)
            print("✓ Step 2: File uploader found")
            
            # ============================================================
            # STEP 3: Create test document and upload
            # ============================================================
            step3 = {"step": 3, "action": "Upload test document", "status": "pending"}
            
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(TEST_DOCUMENT_CONTENT)
                test_file_path = f.name
            
            # Upload the file
            await uploader.set_input_files(test_file_path)
            await page.wait_for_timeout(2000)  # Wait for extraction
            await page.screenshot(path=str(EVIDENCE_DIR / "02_after_upload.png"))
            
            step3["status"] = "pass"
            step3["evidence"] = "02_after_upload.png"
            results["steps"].append(step3)
            print("✓ Step 3: File uploaded")
            
            # ============================================================
            # STEP 4: Verify extraction success message
            # ============================================================
            step4 = {"step": 4, "action": "Verify extraction success", "status": "pending"}
            
            # Look for success message
            success_msg = page.locator("text=Extracted")
            try:
                await expect(success_msg.first).to_be_visible(timeout=10000)
                step4["status"] = "pass"
                print("✓ Step 4: Extraction success message visible")
            except:
                step4["status"] = "fail"
                step4["error"] = "No extraction success message found"
                results["failures"].append(step4)
                print("✗ Step 4: No extraction success message")
            
            results["steps"].append(step4)
            await page.screenshot(path=str(EVIDENCE_DIR / "03_extraction_result.png"))
            
            # ============================================================
            # STEP 5: Click "Continue to Review" or navigate to Configure
            # ============================================================
            step5 = {"step": 5, "action": "Navigate to Configure step", "status": "pending"}
            
            # Try to find and click Continue button
            continue_btn = page.locator("button:has-text('Continue')")
            if await continue_btn.count() > 0:
                await continue_btn.first.click()
                await page.wait_for_timeout(1000)
            
            # If we're on Review, click Continue again
            continue_btn2 = page.locator("button:has-text('Continue')")
            if await continue_btn2.count() > 0:
                await continue_btn2.first.click()
                await page.wait_for_timeout(1000)
            
            await page.screenshot(path=str(EVIDENCE_DIR / "04_configure_step.png"))
            step5["status"] = "pass"
            step5["evidence"] = "04_configure_step.png"
            results["steps"].append(step5)
            print("✓ Step 5: Navigated forward")
            
            # ============================================================
            # STEP 6: THE CRITICAL CHECK - Are extracted values in form?
            # ============================================================
            step6 = {"step": 6, "action": "VERIFY FORM POPULATED", "status": "pending"}
            step6["checks"] = []
            
            await page.wait_for_timeout(2000)  # Let form render
            await page.screenshot(path=str(EVIDENCE_DIR / "05_form_state.png"))
            
            # Check room_length field
            # Streamlit number_input creates an input with aria-label or we find by label text
            
            # Try to find the Room Length input
            room_length_input = page.locator('input[aria-label*="Room Length"]')
            if await room_length_input.count() == 0:
                # Try alternative selectors
                room_length_input = page.locator('input').filter(has_text="45")
            
            # Get all number inputs and check their values
            all_inputs = await page.locator('input[type="number"], input[inputmode="decimal"]').all()
            
            found_values = {}
            for inp in all_inputs:
                try:
                    value = await inp.input_value()
                    aria = await inp.get_attribute("aria-label") or "unknown"
                    found_values[aria] = value
                except:
                    pass
            
            step6["found_inputs"] = found_values
            print(f"   Found input values: {found_values}")
            
            # Check if our expected values are present
            values_found = []
            for expected_val in [45.0, 32.0, 11.0, 1800.0]:
                found = any(str(expected_val) in str(v) or str(int(expected_val)) in str(v) 
                           for v in found_values.values())
                values_found.append((expected_val, found))
                if found:
                    print(f"   ✓ Found expected value: {expected_val}")
                else:
                    print(f"   ✗ MISSING expected value: {expected_val}")
            
            step6["values_found"] = values_found
            
            # Determine pass/fail
            all_found = all(f for _, f in values_found)
            if all_found:
                step6["status"] = "pass"
                print("✓ Step 6: ALL EXTRACTED VALUES FOUND IN FORM")
            else:
                step6["status"] = "FAIL"
                missing = [v for v, f in values_found if not f]
                step6["error"] = f"Missing values in form: {missing}"
                results["failures"].append(step6)
                print(f"✗ Step 6: FAIL - Missing values: {missing}")
            
            results["steps"].append(step6)
            
            # ============================================================
            # FINAL SCREENSHOT - Full page state
            # ============================================================
            await page.screenshot(path=str(EVIDENCE_DIR / "06_final_state.png"), full_page=True)
            
            # Determine overall result
            results["passed"] = len(results["failures"]) == 0
            
        except Exception as e:
            results["error"] = str(e)
            results["passed"] = False
            await page.screenshot(path=str(EVIDENCE_DIR / "error_state.png"))
            print(f"✗ Test failed with error: {e}")
        
        finally:
            # Save trace
            await context.tracing.stop(path=str(EVIDENCE_DIR / "trace.zip"))
            
            # Save console logs
            with open(EVIDENCE_DIR / "console_logs.txt", "w") as f:
                f.write("\n".join(console_logs))
            
            # Save results
            with open(EVIDENCE_DIR / "test_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            await browser.close()
            
            # Cleanup temp file
            Path(test_file_path).unlink(missing_ok=True)
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("HERMENEUTIC UI VERIFICATION - DOCUMENT INGESTION")
    print("=" * 60)
    print(f"\nEvidence directory: {EVIDENCE_DIR}")
    print(f"Target URL: {BASE_URL}")
    print(f"\nExpected values to find in form:")
    for k, v in EXPECTED_VALUES.items():
        print(f"  {k}: {v}")
    print("\n" + "=" * 60)
    
    results = asyncio.run(test_document_ingestion_e2e())
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    if results["passed"]:
        print("\n✅ TEST PASSED - Document ingestion works end-to-end")
    else:
        print("\n❌ TEST FAILED")
        for failure in results.get("failures", []):
            print(f"   - Step {failure.get('step')}: {failure.get('error')}")
    
    print(f"\nEvidence saved to: {EVIDENCE_DIR}")
    print("  - Screenshots: 01_*.png through 06_*.png")
    print("  - Trace: trace.zip")
    print("  - Console logs: console_logs.txt")
    print("  - Results: test_results.json")
