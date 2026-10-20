// Checks encode_bits/decode_bits against the shared packet vectors in tests/vectors
// (see the README there). Needs only src/bch.cpp, not ONNX Runtime or OpenCV:
//
//   g++ -std=c++17 -Icpp/src cpp/tests/conformance.cpp cpp/src/bch.cpp -o conformance
//   ./conformance tests/vectors/packets.tsv

#include "bch.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string& id, const std::string& message) {
    if (!ok) {
        ++failures;
        std::cerr << "FAIL " << id << ": " << message << "\n";
    }
}

// Splits on every separator, keeping empty fields.
std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> fields;
    size_t start = 0;
    for (size_t end; (end = text.find(separator, start)) != std::string::npos; start = end + 1) {
        fields.push_back(text.substr(start, end - start));
    }
    fields.push_back(text.substr(start));
    return fields;
}

// encode <id> <schema> <payload> <packet> <python> <rust> <cpp>
void check_encode(const std::vector<std::string>& f) {
    const std::string& id = f[1];
    const std::string& expect = f[7];
    std::string packet;
    bool threw = false;
    try {
        packet = trustmark::encode_bits(f[3], f[2]);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    if (expect == "packet") {
        check(!threw && packet == f[4], id, "expected " + f[4] + ", got " + (threw ? "an exception" : packet));
    } else if (expect.rfind("error:", 0) == 0) {
        check(threw, id, "expected an exception, got " + packet);
    } else {
        check(false, id, "unknown expectation " + expect);
    }
}

// decode <id> <input> <schema> <data> <valid_under> <python> <rust> <cpp> <js>
void check_decode(const std::vector<std::string>& f) {
    const std::string& id = f[1];
    const std::string& expect = f[8];
    trustmark::Decoded decoded;
    const bool valid = trustmark::decode_bits(f[2], decoded);
    const std::string found = decoded.version + "=" + decoded.bits;
    if (valid) {
        bool listed = false;
        for (const auto& entry : split(f[5], ';')) listed = listed || entry == found;
        check(listed, id, found + " is not listed in valid_under");
    }
    const bool original = valid && decoded.version == f[3] && decoded.bits == f[4];
    if (expect == "valid") {
        check(original, id, "expected " + f[3] + "=" + f[4] + ", got " + (valid ? found : "invalid"));
    } else if (expect == "invalid") {
        check(!valid, id, "expected invalid, got " + found);
    } else if (expect == "not_original") {
        check(!original, id, "returned the original payload");
    } else {
        check(false, id, "unknown expectation " + expect);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "tests/vectors/packets.tsv";
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        return 2;
    }
    int encoded = 0, decoded = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto fields = split(line, '\t');
        if (fields[0] == "encode" && fields.size() == 8) {
            check_encode(fields);
            ++encoded;
        } else if (fields[0] == "decode" && fields.size() == 10) {
            check_decode(fields);
            ++decoded;
        } else {
            check(false, fields.size() > 1 ? fields[1] : line, "malformed vector");
        }
    }
    if (encoded == 0 || decoded == 0) {
        std::cerr << "no vectors found in " << path << "\n";
        return 1;
    }
    std::cout << encoded << " encode and " << decoded << " decode vectors checked, " << failures
              << " failures\n";
    return failures ? 1 : 0;
}
