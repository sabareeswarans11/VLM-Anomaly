import XCTest
@testable import VLMAnomalyEdge

// Parser parity tests: mirrors the golden cases in tests/fixtures/ and
// the five-step fallback chain in src/vlm_anomaly/utils/json_parsing.py.
final class ParserTests: XCTestCase {

    let parser = AnomalyResponseParser.shared

    // MARK: - Step 1: direct JSON

    func testDirectJSON_anomalous() {
        let raw = #"{"is_anomalous": true, "confidence": 0.92, "description": "Crack on surface", "defect_type": "crack"}"#
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertEqual(d["is_anomalous"] as? Bool, true)
        XCTAssertEqual(d["confidence"] as? Double ?? 0, 0.92, accuracy: 0.001)
        XCTAssertEqual(d["description"] as? String, "Crack on surface")
        XCTAssertEqual(d["defect_type"] as? String, "crack")
    }

    func testDirectJSON_normal() {
        let raw = #"{"is_anomalous": false, "confidence": 0.05, "description": "No defects visible."}"#
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertEqual(d["is_anomalous"] as? Bool, false)
    }

    // MARK: - Step 2: embedded JSON block

    func testEmbeddedBlock() {
        let raw = "Sure! Here is my analysis: {\"is_anomalous\": true, \"confidence\": 0.8} Hope that helps."
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertEqual(d["is_anomalous"] as? Bool, true)
    }

    // MARK: - Step 3: code fences

    func testCodeFence_json() {
        let raw = "```json\n{\"is_anomalous\": false, \"confidence\": 0.1}\n```"
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertEqual(d["is_anomalous"] as? Bool, false)
    }

    func testCodeFence_plain() {
        let raw = "```\n{\"is_anomalous\": true, \"confidence\": 0.95}\n```"
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertEqual(d["is_anomalous"] as? Bool, true)
    }

    // MARK: - Step 4: regex fallback

    func testRegexFallback_anomalous() {
        let raw = #"The image shows "is_anomalous": true and "confidence": 0.75 from inspection."#
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertEqual(d["is_anomalous"] as? Bool, true)
        XCTAssertEqual(d["confidence"] as? Double ?? 0, 0.75, accuracy: 0.001)
    }

    // MARK: - Step 5: parse error

    func testParseError_garbage() {
        let (_, err) = parser.parse("This is not JSON at all.")
        XCTAssertTrue(err)
    }

    func testParseError_empty() {
        let (_, err) = parser.parse("")
        XCTAssertTrue(err)
    }

    // MARK: - Normalisation

    func testConfidenceClamp() {
        let raw = #"{"is_anomalous": true, "confidence": 1.5}"#
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertLessThanOrEqual(d["confidence"] as? Double ?? 2, 1.0)
    }

    func testStringBoolean() {
        let raw = #"{"is_anomalous": "true", "confidence": 0.9}"#
        let (d, err) = parser.parse(raw)
        XCTAssertFalse(err)
        XCTAssertEqual(d["is_anomalous"] as? Bool, true)
    }

    func testMissingIsAnomalous() {
        let raw = #"{"confidence": 0.5, "description": "something"}"#
        let (d, err) = parser.parse(raw)
        // normalise() returns [:] when is_anomalous is missing → treat as parse error
        XCTAssertTrue(err || d.isEmpty)
    }
}
