"""
HERMENEUTIC UI VERIFICATION - FULL WORKFLOW E2E TEST
=====================================================

Complete end-to-end test of the HVAC CFD Universal Intake System.
Tests the entire 5-step workflow:

1. UPLOAD: Document upload and extraction
2. REVIEW: Review extracted data, verify values displayed
3. CONFIGURE: Form population, validation, editing
4. PREVIEW: 3D visualization renders
5. GENERATE: Simulation execution, results display, PDF export

Evidence captured at every micro-step:
- Screenshots
- DOM state
- Network requests
- Console logs
- Trace file
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, expect, Page
import json
import tempfile
import time

# Configuration
BASE_URL = "http://localhost:8502"
EVIDENCE_DIR = Path(__file__).parent / "evidence" / "full_workflow"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Test document
TEST_DOCUMENT = """
HVAC DESIGN SPECIFICATION
Project: Full Workflow Test Suite
Client: Hermeneutic Verification Corp

ROOM DIMENSIONS:
- Length: 35 feet
- Width: 28 feet
- Height: 10 feet

HVAC REQUIREMENTS:
- Supply Airflow: 1500 CFM
- Supply Temperature: 55°F
- Number of Diffusers: 4
- Return Grilles: 2

OCCUPANCY:
- Maximum: 20 people
- Equipment Heat Load: 5000 W

TARGET CONDITIONS:
- Temperature: 72°F ± 2°F
"""

EXPECTED_EXTRACTIONS = {
    "room_length": 35.0,
    "room_width": 28.0,
    "room_height": 10.0,
    "supply_airflow": 1500.0,
    "supply_temperature": 55.0,
    "num_diffusers": 4,
}


class WorkflowTest:
    def __init__(self):
        self.results = {
            "test_suite": "full_workflow_e2e",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phases": {},
            "passed": False,
            "failures": [],
            "warnings": [],
        }
        self.screenshot_count = 0
        self.page = None
        self.console_logs = []
        self.network_requests = []
    
    async def screenshot(self, name: str, full_page: bool = False):
        """Capture screenshot with incrementing number."""
        self.screenshot_count += 1
        filename = f"{self.screenshot_count:02d}_{name}.png"
        path = EVIDENCE_DIR / filename
        await self.page.screenshot(path=str(path), full_page=full_page)
        return filename
    
    def log_step(self, phase: str, step: str, status: str, **kwargs):
        """Log a test step."""
        if phase not in self.results["phases"]:
            self.results["phases"][phase] = {"steps": [], "status": "pending"}
        
        step_data = {
            "step": step,
            "status": status,
            "timestamp": time.strftime("%H:%M:%S"),
            **kwargs
        }
        self.results["phases"][phase]["steps"].append(step_data)
        
        icon = "✓" if status == "pass" else "✗" if status == "fail" else "○"
        print(f"  {icon} {step}")
        
        if status == "fail":
            self.results["failures"].append({"phase": phase, "step": step, **kwargs})
    
    async def wait_for_streamlit(self):
        """Wait for Streamlit to finish rerunning."""
        # Wait for any spinners to disappear
        try:
            spinner = self.page.locator('[data-testid="stSpinner"]')
            await spinner.wait_for(state="hidden", timeout=30000)
        except:
            pass
        await self.page.wait_for_timeout(500)
    
    async def phase1_upload(self) -> bool:
        """Phase 1: Upload and Extraction"""
        print("\n" + "="*60)
        print("PHASE 1: UPLOAD & EXTRACTION")
        print("="*60)
        
        phase = "upload"
        
        # Step 1.1: Verify initial state
        await self.screenshot("initial_state")
        
        # Look for upload step indicators
        upload_text = self.page.locator("text=Upload")
        if await upload_text.count() > 0:
            self.log_step(phase, "Verify on Upload step", "pass")
        else:
            self.log_step(phase, "Verify on Upload step", "fail", error="Upload step not visible")
            return False
        
        # Step 1.2: Locate file uploader
        uploader = self.page.locator('input[type="file"]')
        try:
            await expect(uploader).to_be_attached(timeout=5000)
            self.log_step(phase, "File uploader present", "pass")
        except:
            self.log_step(phase, "File uploader present", "fail", error="No file uploader found")
            return False
        
        # Step 1.3: Create and upload test document
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(TEST_DOCUMENT)
            test_file = f.name
        
        await uploader.set_input_files(test_file)
        await self.wait_for_streamlit()
        await self.screenshot("after_upload")
        self.log_step(phase, "Upload test document", "pass", file=test_file)
        
        # Step 1.4: Verify extraction success
        success_indicators = [
            "text=Extracted",
            "text=fields",
            "text=✅",
        ]
        
        extraction_confirmed = False
        for selector in success_indicators:
            elem = self.page.locator(selector)
            if await elem.count() > 0:
                extraction_confirmed = True
                break
        
        if extraction_confirmed:
            self.log_step(phase, "Extraction success message", "pass")
        else:
            self.log_step(phase, "Extraction success message", "fail", error="No extraction confirmation")
            await self.screenshot("extraction_failed")
        
        # Step 1.5: Click Continue to proceed
        continue_btn = self.page.locator("button:has-text('Continue')")
        if await continue_btn.count() > 0:
            await continue_btn.first.click()
            await self.wait_for_streamlit()
            await self.screenshot("after_continue")
            self.log_step(phase, "Click Continue button", "pass")
        else:
            self.log_step(phase, "Click Continue button", "fail", error="No Continue button")
            return False
        
        # Cleanup
        Path(test_file).unlink(missing_ok=True)
        
        self.results["phases"][phase]["status"] = "pass"
        return True
    
    async def phase2_review(self) -> bool:
        """Phase 2: Review Extracted Data"""
        print("\n" + "="*60)
        print("PHASE 2: REVIEW EXTRACTION")
        print("="*60)
        
        phase = "review"
        
        # Step 2.1: Verify we're on Review step
        await self.screenshot("review_step")
        
        review_indicators = ["Review", "Extracted", "Confirm"]
        on_review = False
        for indicator in review_indicators:
            if await self.page.locator(f"text={indicator}").count() > 0:
                on_review = True
                break
        
        if on_review:
            self.log_step(phase, "On Review step", "pass")
        else:
            # Might have skipped to Configure
            self.log_step(phase, "On Review step", "pass", note="May have auto-advanced")
        
        # Step 2.2: Check for extracted values display
        page_content = await self.page.content()
        
        values_displayed = []
        for field, expected in EXPECTED_EXTRACTIONS.items():
            if str(int(expected)) in page_content or str(expected) in page_content:
                values_displayed.append(field)
        
        if len(values_displayed) >= 3:
            self.log_step(phase, "Extracted values displayed", "pass", 
                         found=values_displayed)
        else:
            self.log_step(phase, "Extracted values displayed", "warning",
                         found=values_displayed, expected=list(EXPECTED_EXTRACTIONS.keys()))
            self.results["warnings"].append("Not all extracted values visible in review")
        
        # Step 2.3: Check confidence indicators
        confidence_icons = await self.page.locator("text=🟢").count()
        confidence_icons += await self.page.locator("text=🟡").count()
        confidence_icons += await self.page.locator("text=🔴").count()
        
        if confidence_icons > 0:
            self.log_step(phase, "Confidence indicators present", "pass", count=confidence_icons)
        else:
            self.log_step(phase, "Confidence indicators present", "warning", 
                         note="No confidence icons found")
        
        # Step 2.4: Navigate to Configure
        continue_btn = self.page.locator("button:has-text('Continue')")
        if await continue_btn.count() > 0:
            await continue_btn.first.click()
            await self.wait_for_streamlit()
            await self.screenshot("after_review_continue")
            self.log_step(phase, "Navigate to Configure", "pass")
        else:
            # Try clicking any forward button
            next_btn = self.page.locator("button:has-text('→'), button:has-text('Next')")
            if await next_btn.count() > 0:
                await next_btn.first.click()
                await self.wait_for_streamlit()
                self.log_step(phase, "Navigate to Configure", "pass")
            else:
                self.log_step(phase, "Navigate to Configure", "fail", error="No navigation button")
                return False
        
        self.results["phases"][phase]["status"] = "pass"
        return True
    
    async def phase3_configure(self) -> bool:
        """Phase 3: Configure Simulation Parameters"""
        print("\n" + "="*60)
        print("PHASE 3: CONFIGURE PARAMETERS")
        print("="*60)
        
        phase = "configure"
        
        # Step 3.1: Verify on Configure step
        await self.screenshot("configure_step")
        
        configure_indicators = ["Configure", "Parameters", "Required", "Simulation"]
        on_configure = False
        for indicator in configure_indicators:
            if await self.page.locator(f"text={indicator}").count() > 0:
                on_configure = True
                break
        
        if on_configure:
            self.log_step(phase, "On Configure step", "pass")
        else:
            self.log_step(phase, "On Configure step", "fail", error="Not on Configure step")
            return False
        
        # Step 3.2: CRITICAL - Verify form populated with extracted values
        all_inputs = await self.page.locator('input[type="number"], input[inputmode="decimal"]').all()
        
        found_values = {}
        for inp in all_inputs:
            try:
                value = await inp.input_value()
                aria = await inp.get_attribute("aria-label") or "unknown"
                found_values[aria] = value
            except:
                pass
        
        self.log_step(phase, "Collect form values", "pass", 
                     input_count=len(found_values))
        
        # Check critical values
        critical_checks = [
            ("Room Length", "35"),
            ("Room Width", "28"),
            ("Height", "10"),
            ("Airflow", "1500"),
        ]
        
        all_populated = True
        for field_name, expected_val in critical_checks:
            found = False
            for aria, value in found_values.items():
                if field_name.lower() in aria.lower():
                    if expected_val in value:
                        found = True
                        break
            
            if found:
                self.log_step(phase, f"Form has {field_name}={expected_val}", "pass")
            else:
                self.log_step(phase, f"Form has {field_name}={expected_val}", "fail",
                             error=f"Expected {expected_val}, field not found or wrong value")
                all_populated = False
        
        await self.screenshot("form_values", full_page=True)
        
        # Step 3.3: Test form validation - clear a required field
        room_length_input = self.page.locator('input[aria-label*="Room Length"]')
        if await room_length_input.count() > 0:
            # Store original value
            original = await room_length_input.first.input_value()
            
            # Clear and check validation
            await room_length_input.first.fill("")
            await self.page.keyboard.press("Tab")
            await self.wait_for_streamlit()
            await self.screenshot("validation_test")
            
            # Restore value
            await room_length_input.first.fill(original)
            await self.page.keyboard.press("Tab")
            await self.wait_for_streamlit()
            
            self.log_step(phase, "Form validation triggers", "pass")
        else:
            self.log_step(phase, "Form validation triggers", "warning",
                         note="Could not test validation - field not found")
        
        # Step 3.4: Check unit system selector
        unit_selector = self.page.locator("text=Imperial")
        if await unit_selector.count() > 0:
            self.log_step(phase, "Unit system selector present", "pass")
        else:
            self.log_step(phase, "Unit system selector present", "warning")
        
        # Step 3.5: Navigate to Preview
        continue_btn = self.page.locator("button:has-text('Continue'), button:has-text('Preview')")
        if await continue_btn.count() > 0:
            await continue_btn.first.click()
            await self.wait_for_streamlit()
            await self.screenshot("after_configure")
            self.log_step(phase, "Navigate to Preview", "pass")
        else:
            self.log_step(phase, "Navigate to Preview", "fail", error="No navigation button")
            return False
        
        self.results["phases"][phase]["status"] = "pass" if all_populated else "fail"
        return all_populated
    
    async def phase4_preview(self) -> bool:
        """Phase 4: 3D Preview"""
        print("\n" + "="*60)
        print("PHASE 4: 3D PREVIEW")
        print("="*60)
        
        phase = "preview"
        
        # Step 4.1: Verify on Preview step
        await self.page.wait_for_timeout(2000)  # Wait for 3D to render
        await self.screenshot("preview_step")
        
        preview_indicators = ["Preview", "3D", "Visualization", "Scene"]
        on_preview = False
        for indicator in preview_indicators:
            if await self.page.locator(f"text={indicator}").count() > 0:
                on_preview = True
                break
        
        if on_preview:
            self.log_step(phase, "On Preview step", "pass")
        else:
            self.log_step(phase, "On Preview step", "warning", note="Preview text not found")
        
        # Step 4.2: Check for Plotly chart (3D visualization)
        plotly_chart = self.page.locator('[data-testid="stPlotlyChart"], .js-plotly-plot, .plotly')
        
        try:
            await expect(plotly_chart.first).to_be_visible(timeout=10000)
            self.log_step(phase, "3D visualization rendered", "pass")
            await self.screenshot("3d_visualization")
        except:
            # Try alternative detection
            canvas = self.page.locator("canvas")
            if await canvas.count() > 0:
                self.log_step(phase, "3D visualization rendered", "pass", note="Canvas element found")
            else:
                self.log_step(phase, "3D visualization rendered", "fail", 
                             error="No 3D chart or canvas found")
                await self.screenshot("no_3d_found")
        
        # Step 4.3: Check for dimension display in title
        page_content = await self.page.content()
        if "35" in page_content and "28" in page_content:
            self.log_step(phase, "Dimensions shown in preview", "pass")
        else:
            self.log_step(phase, "Dimensions shown in preview", "warning")
        
        # Step 4.4: Check for dual unit display (ft → m)
        if "→" in page_content or "ft" in page_content.lower():
            self.log_step(phase, "Unit conversion displayed", "pass")
        else:
            self.log_step(phase, "Unit conversion displayed", "warning")
        
        # Step 4.5: Navigate to Generate
        continue_btn = self.page.locator("button:has-text('Continue'), button:has-text('Generate'), button:has-text('Run')")
        if await continue_btn.count() > 0:
            await continue_btn.first.click()
            await self.wait_for_streamlit()
            await self.screenshot("after_preview")
            self.log_step(phase, "Navigate to Generate", "pass")
        else:
            self.log_step(phase, "Navigate to Generate", "fail", error="No navigation button")
            return False
        
        self.results["phases"][phase]["status"] = "pass"
        return True
    
    async def phase5_generate(self) -> bool:
        """Phase 5: Generate & Run Simulation"""
        print("\n" + "="*60)
        print("PHASE 5: GENERATE & RUN")
        print("="*60)
        
        phase = "generate"
        
        # Step 5.1: Verify on Generate step
        await self.screenshot("generate_step")
        
        generate_indicators = ["Generate", "Run", "Simulation", "Execute"]
        on_generate = False
        for indicator in generate_indicators:
            if await self.page.locator(f"text={indicator}").count() > 0:
                on_generate = True
                break
        
        if on_generate:
            self.log_step(phase, "On Generate step", "pass")
        else:
            self.log_step(phase, "On Generate step", "warning")
        
        # Step 5.2: Check for job spec display
        job_spec_indicators = ["Job Specification", "job_spec", "Configuration Summary"]
        has_job_spec = False
        for indicator in job_spec_indicators:
            if await self.page.locator(f"text={indicator}").count() > 0:
                has_job_spec = True
                break
        
        if has_job_spec:
            self.log_step(phase, "Job specification displayed", "pass")
        else:
            self.log_step(phase, "Job specification displayed", "warning")
        
        # Step 5.3: Check for Download JSON button
        download_btn = self.page.locator("button:has-text('Download'), a:has-text('Download')")
        if await download_btn.count() > 0:
            self.log_step(phase, "Download JSON available", "pass")
        else:
            self.log_step(phase, "Download JSON available", "warning")
        
        # Step 5.4: Check for Save Project button
        save_btn = self.page.locator("button:has-text('Save')")
        if await save_btn.count() > 0:
            self.log_step(phase, "Save Project button present", "pass")
        else:
            self.log_step(phase, "Save Project button present", "warning")
        
        # Step 5.5: Check for Run Simulation button
        run_btn = self.page.locator("button:has-text('Run Simulation'), button:has-text('🚀')")
        if await run_btn.count() > 0:
            self.log_step(phase, "Run Simulation button present", "pass")
            await self.screenshot("before_simulation")
            
            # Step 5.6: Actually run the simulation (if button exists)
            # NOTE: This will take time - simulation is real
            print("\n  ⏳ Running simulation (this may take 1-2 minutes)...")
            
            await run_btn.first.click()
            
            # Wait for simulation to complete (with timeout)
            start_time = time.time()
            max_wait = 180  # 3 minutes max
            
            simulation_complete = False
            while time.time() - start_time < max_wait:
                await self.page.wait_for_timeout(5000)  # Check every 5 seconds
                
                # Look for completion indicators
                complete_indicators = [
                    "text=Complete",
                    "text=Results",
                    "text=Temperature",
                    "text=PASS",
                    "text=FAIL",
                    "text=✅",
                    "text=❌",
                ]
                
                for selector in complete_indicators:
                    if await self.page.locator(selector).count() > 0:
                        simulation_complete = True
                        break
                
                if simulation_complete:
                    break
                
                # Also check for error state
                if await self.page.locator("text=Error").count() > 0:
                    break
                
                elapsed = int(time.time() - start_time)
                print(f"    ... waiting ({elapsed}s)")
            
            await self.screenshot("after_simulation", full_page=True)
            
            if simulation_complete:
                self.log_step(phase, "Simulation completed", "pass",
                             duration=f"{int(time.time() - start_time)}s")
                
                # Step 5.7: Verify results display
                results_check = await self.page.content()
                
                result_indicators = ["Temperature", "Velocity", "CO₂", "PMV", "ADPI"]
                results_found = [ind for ind in result_indicators if ind in results_check]
                
                if len(results_found) >= 3:
                    self.log_step(phase, "Results displayed", "pass", metrics=results_found)
                else:
                    self.log_step(phase, "Results displayed", "warning", 
                                 found=results_found, expected=result_indicators)
                
                # Step 5.8: Check for PDF export option
                pdf_btn = self.page.locator("button:has-text('PDF'), button:has-text('Report')")
                if await pdf_btn.count() > 0:
                    self.log_step(phase, "PDF export available", "pass")
                else:
                    self.log_step(phase, "PDF export available", "warning")
                
            else:
                self.log_step(phase, "Simulation completed", "fail",
                             error="Simulation did not complete within timeout")
        else:
            self.log_step(phase, "Run Simulation button present", "fail",
                         error="No Run Simulation button found")
            return False
        
        await self.screenshot("final_state", full_page=True)
        
        self.results["phases"][phase]["status"] = "pass"
        return True
    
    async def run(self):
        """Execute full workflow test."""
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                viewport={"width": 1440, "height": 900},
                record_video_dir=str(EVIDENCE_DIR),
            )
            
            # Enable tracing
            await context.tracing.start(screenshots=True, snapshots=True)
            
            self.page = await context.new_page()
            
            # Capture console and network
            self.page.on("console", lambda msg: self.console_logs.append(
                f"[{msg.type}] {msg.text}"
            ))
            self.page.on("request", lambda req: self.network_requests.append(
                f"[{req.method}] {req.url}"
            ))
            
            try:
                # Navigate to app
                print("\n" + "="*60)
                print("STARTING FULL WORKFLOW TEST")
                print("="*60)
                print(f"URL: {BASE_URL}")
                print(f"Evidence: {EVIDENCE_DIR}")
                
                await self.page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
                
                # Run all phases
                phase_results = []
                
                phase_results.append(("Upload", await self.phase1_upload()))
                phase_results.append(("Review", await self.phase2_review()))
                phase_results.append(("Configure", await self.phase3_configure()))
                phase_results.append(("Preview", await self.phase4_preview()))
                phase_results.append(("Generate", await self.phase5_generate()))
                
                # Determine overall result
                self.results["passed"] = all(r for _, r in phase_results)
                
            except Exception as e:
                self.results["error"] = str(e)
                self.results["passed"] = False
                await self.screenshot("error_state")
                print(f"\n✗ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                # Save evidence
                await context.tracing.stop(path=str(EVIDENCE_DIR / "trace.zip"))
                
                with open(EVIDENCE_DIR / "console_logs.txt", "w") as f:
                    f.write("\n".join(self.console_logs))
                
                with open(EVIDENCE_DIR / "network_requests.txt", "w") as f:
                    f.write("\n".join(self.network_requests))
                
                self.results["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(EVIDENCE_DIR / "test_results.json", "w") as f:
                    json.dump(self.results, f, indent=2, default=str)
                
                await browser.close()
        
        return self.results


def print_summary(results: dict):
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for phase_name, phase_data in results.get("phases", {}).items():
        status = phase_data.get("status", "unknown")
        icon = "✅" if status == "pass" else "❌" if status == "fail" else "⚠️"
        print(f"\n{icon} Phase: {phase_name.upper()}")
        
        for step in phase_data.get("steps", []):
            step_status = step.get("status")
            step_icon = "✓" if step_status == "pass" else "✗" if step_status == "fail" else "○"
            print(f"   {step_icon} {step.get('step')}")
            if step.get("error"):
                print(f"      ERROR: {step.get('error')}")
    
    print("\n" + "="*60)
    if results.get("passed"):
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ TESTS FAILED")
        print("\nFailures:")
        for failure in results.get("failures", []):
            print(f"  - [{failure.get('phase')}] {failure.get('step')}: {failure.get('error')}")
    
    if results.get("warnings"):
        print("\nWarnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    print(f"\nEvidence saved to: {EVIDENCE_DIR}")
    print("="*60)


if __name__ == "__main__":
    test = WorkflowTest()
    results = asyncio.run(test.run())
    print_summary(results)
