//
// Created by Calvin Giles on 22/03/2018.
//

#ifndef ZIMMER_EXPECT_H
#define ZIMMER_EXPECT_H


template<typename InputType, typename OutputType, typename ParamType>
class Expect {
public:
    /**
     * Explicit constructor
     * @param input - TimeSeries to be used as test input
     * @param expected - expected return value from test call
     * @param expect_reason - reason why the return value is the expected behaviour
     * @param value - parameter to pass to filter method
     */
    Expect(
            const std::string &expect_reason,
            const InputType &input,
            const OutputType &expected,
            const ParamType &value
    ) : expect_reason(expect_reason), input(input), expected(expected), value(value) {}

    /**
     * Delegating constructor that initialises value to it's default
     * @param input - TimeSeries to be used as test input
     * @param expected - expected return value from test call
     * @param expect_reason - reason why the return value is the expected behaviour
     */
    Expect(
            const std::string &expect_reason,
            const InputType &input,
            const OutputType &expected
    ) : expect_reason(expect_reason), input(input), expected(expected), value() {}

    const std::string expect_reason;
    const InputType input;
    const OutputType expected;
    const ParamType value;
};


#endif //ZIMMER_EXPECT_H
