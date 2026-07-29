#include "moai/openfhe/approximation_registry.hpp"

#include "math/chebyshev.h"
#include "utils/hashutil.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace moai::openfhe {
namespace {

void RefreshCoefficientHash(ApproximationContract& contract) {
    contract.coefficient_sha256 =
        ComputeCoefficientSha256(contract.coefficients);
}

ApproximationContract MakeChebyshevContract(
    std::string contract_id,
    std::function<double(double)> function,
    double minimum,
    double maximum,
    uint32_t degree,
    uint32_t required_depth,
    uint64_t estimated_multiplications) {
    ApproximationContract contract;
    contract.contract_id = std::move(contract_id);
    contract.basis = "chebyshev";
    contract.interval = {minimum, maximum};
    contract.degree = degree;
    contract.required_depth = required_depth;
    contract.estimated_multiplications = estimated_multiplications;
    contract.coefficients = lbcrypto::EvalChebyshevCoefficients(
        function,
        minimum,
        maximum,
        degree);
    RefreshCoefficientHash(contract);
    return contract;
}

double EvaluateChebyshevAt(
    const ApproximationContract& contract,
    double input) {
    const double normalized =
        (2.0 * input - contract.interval.minimum - contract.interval.maximum) /
        (contract.interval.maximum - contract.interval.minimum);
    double previous = 1.0;
    double current = normalized;
    double result = contract.coefficients[0] / 2.0;
    if (contract.degree >= 1) {
        result += contract.coefficients[1] * current;
    }
    for (uint32_t degree = 2; degree <= contract.degree; ++degree) {
        const double next = 2.0 * normalized * current - previous;
        result += contract.coefficients[degree] * next;
        previous = current;
        current = next;
    }
    return result;
}

void ConstrainZeroAtOrigin(ApproximationContract& contract) {
    if (contract.interval.minimum > 0.0 || contract.interval.maximum < 0.0) {
        throw std::logic_error(
            "zero-at-origin constraint requires an interval containing zero");
    }
    const double raw_at_origin = EvaluateChebyshevAt(contract, 0.0);
    contract.coefficients[0] -= 2.0 * raw_at_origin;
    if (std::abs(EvaluateChebyshevAt(contract, 0.0)) > 1e-12) {
        throw std::logic_error(
            "zero-at-origin Chebyshev correction exceeded its oracle gate");
    }
    RefreshCoefficientHash(contract);
}

}  // namespace

std::string ComputeCoefficientSha256(
    const std::vector<double>& coefficients) {
    static_assert(
        sizeof(double) == sizeof(uint64_t),
        "coefficient hashing requires 64-bit doubles");
    static_assert(
        std::numeric_limits<double>::is_iec559,
        "coefficient hashing requires IEEE-754 doubles");
    std::string encoded;
    encoded.reserve(coefficients.size() * sizeof(double));
    for (const double coefficient : coefficients) {
        uint64_t bits = 0;
        std::memcpy(&bits, &coefficient, sizeof(bits));
        for (uint32_t byte = 0; byte < sizeof(bits); ++byte) {
            encoded.push_back(static_cast<char>((bits >> (8 * byte)) & 0xff));
        }
    }
    return lbcrypto::HashUtil::HashString(std::move(encoded));
}

double PaperCompatSoftmaxShiftContract::At(
    std::size_t layer,
    std::size_t head) const {
    if (contract_id != kPaperCompatSoftmaxShiftContractId ||
        values_sha256 != kPaperCompatSoftmaxShiftSha256 ||
        layer_count != kPaperCompatEncoderLayers ||
        head_count != kPaperCompatAttentionHeads ||
        values.size() != layer_count * head_count ||
        ComputeCoefficientSha256(values) != kPaperCompatSoftmaxShiftSha256) {
        throw std::logic_error(
            "paper_compat Softmax public-shift registry drifted");
    }
    if (layer >= layer_count || head >= head_count) {
        throw std::out_of_range(
            "paper_compat Softmax layer/head index is out of range");
    }
    return values[layer * head_count + head];
}

PaperCompatSoftmaxShiftContract MakePaperCompatSoftmaxShiftContract() {
    PaperCompatSoftmaxShiftContract contract;
    contract.contract_id = kPaperCompatSoftmaxShiftContractId;
    contract.values_sha256 = kPaperCompatSoftmaxShiftSha256;
    contract.layer_count = kPaperCompatEncoderLayers;
    contract.head_count = kPaperCompatAttentionHeads;
    contract.values = {
        2.6955842567519865, 3.6057040939810996, 5.435268151971699,
        5.3354395970933108, 2.3525915308218086, 3.2975371068394495,
        2.5869569241823589, 3.2336628300139187, 3.3292014333817992,
        2.6114129606671788, 5.6043151058842371, 4.7935705649462097,
        3.6584036828045128, 5.8303078684135041, 2.9766493219191048,
        3.6019507049829427, 5.8526831925488203, 4.2024193163318522,
        7.0656032460077141, 4.0657396082428168, 2.779362527815032,
        2.507358119816991, 4.8904426240943284, 4.5066689440716328,
        8.8276451406474017, 5.6606411788982847, 3.800839758088105,
        2.7550667644819269, 3.7824411289243702, 4.2625434793803736,
        4.6584931945460069, 3.9784442979630068, 3.9574819726140515,
        9.1520096755192668, 4.0210392038958664, 5.9242706980062447,
        9.9305396465335498, 4.3003959592600562, 4.9795424582404149,
        5.1930189371900077, 4.4887222317914581, 6.4130820622972857,
        4.4322770490418248, 3.3026237250599237, 4.387398175677772,
        6.5850128243023667, 5.2283353448943028, 4.687135256967915,
        5.3145443584041789, 3.2575300515987853, 3.1973916882540134,
        6.7084375864158403, 5.4580789503024256, 4.9035728294899608,
        5.3245795030890211, 5.4043222405931779, 3.725639033803346,
        4.0839637821240906, 4.4097447931442506, 4.2461811809204297,
        4.1090674125208748, 5.6323440605954138, 3.5038968165913298,
        4.6262458094056491, 3.4125351414161429, 5.0214096078185753,
        5.1366667623739986, 4.9946615709590381, 4.216565928606637,
        5.2359471522870518, 3.9576008024129523, 4.6515611631621301,
        6.4734052563165978, 4.7097484627912483, 4.1665554134100367,
        5.7842448400382462, 5.0801756628015111, 4.7684185776810173,
        3.3666313374002379, 4.224068526108482, 3.4419765244492186,
        4.2524564490884007, 5.9242502271206643, 6.8584134794842599,
        4.8688426824949698, 3.7491252391666396, 4.2710862848402797,
        5.6999236316529291, 6.0703399761793433, 4.757467343428182,
        4.8604662534052459, 5.8592209384305711, 4.2447740097209827,
        4.9783045195891731, 4.3707716915473815, 5.4740361447145931,
        4.618264907707438, 3.4673225067379874, 6.0861907619363027,
        4.7394869387486533, 5.1758412362414097, 4.4444253170544936,
        5.1826681148414302, 3.3391241962576705, 3.876188033588797,
        3.5988230100648675, 5.1301702572720451, 4.9169634450469415,
        5.7636052277543355, 4.1874269541351552, 3.6324090007020855,
        3.8166332654099868, 3.9990962848588176, 3.057362665853443,
        4.1286064843702199, 5.3450702039518516, 4.0810116633217417,
        4.2525679223681703, 3.8537636123576586, 2.435433577032053,
        2.7072358980072089, 4.8516652018908761, 3.8280614159583566,
        1.5300748166886136, 3.9531441723105685, 4.4757090258805459,
        3.9049836622112979, 2.201120701835988, 2.8501437749948915,
        5.7294977870159753, 5.8776156607967769, 3.8324580072429812,
        2.0393165828259554, 2.5209592387704478, 2.0863472222229373,
        3.4778119283651474, 3.0855074037588497, 4.03473829797192,
        2.3124879330084456, 2.3241630267483671, 4.8474311348930366,
        3.3094130852983952, 2.8106675552754474, 2.4947812043033117,
    };
    static_cast<void>(contract.At(0, 0));
    return contract;
}

double PaperCompatSoftmaxPublicShift(
    std::size_t layer,
    std::size_t head) {
    static const auto contract = MakePaperCompatSoftmaxShiftContract();
    return contract.At(layer, head);
}

PaperCompatNonlinearContracts MakePaperCompatNonlinearContracts() {
    PaperCompatNonlinearContracts contracts;
    contracts.softmax_shifts = MakePaperCompatSoftmaxShiftContract();
    contracts.gelu = MakeChebyshevContract(
        "gelu_d319_zero_at_origin",
        [](double value) {
            return 0.5 * value *
                (1.0 + std::erf(value / std::sqrt(2.0)));
        },
        -80.0,
        128.0,
        319,
        10,
        33);
    ConstrainZeroAtOrigin(contracts.gelu);
    contracts.softmax_exponential = MakeChebyshevContract(
        "softmax_exp_d27",
        [](double value) { return std::exp(value); },
        -16.0,
        5.0,
        27,
        6,
        10);
    contracts.softmax_reciprocal = MakeChebyshevContract(
        "softmax_reciprocal_d383",
        [](double value) { return 1.0 / value; },
        0.01,
        80.0,
        383,
        10,
        35);
    contracts.layernorm_inverse_sqrt = MakeChebyshevContract(
        "layernorm_inverse_sqrt_d159",
        [](double value) { return 1.0 / std::sqrt(value); },
        0.5,
        1536.0,
        159,
        9,
        23);
    return contracts;
}

}  // namespace moai::openfhe
