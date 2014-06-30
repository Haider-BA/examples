#include "Vector3d.h"
#include "gtest/gtest.h"

namespace
{

// The fixture for testing class Vector3d.
class Vector3dTest : public ::testing::Test
{
protected:
	// You can remove any or all of the following functions if its body
	// is empty.

	Vector3dTest()
	{
		// You can do set-up work for each test here.
		a.set(1, 1, 0);
		b.set(1, 1, 1);
	}

	virtual ~Vector3dTest()
	{
		// You can do clean-up work that doesn't throw exceptions here.
	}

	// If the constructor and destructor are not enough for setting up
	// and cleaning up each test, you can define the following methods:

	virtual void SetUp()
	{
		// Code here will be called immediately after the constructor (right
		// before each test).
	}

	virtual void TearDown()
	{
		// Code here will be called immediately after each test (right
		// before the destructor).
	}

	// Objects declared here can be used by all tests in the test case for Foo.
	Vector3d a, b;
};

// Tests that a method does what it's supposed to.
TEST_F(Vector3dTest, Magnitude)
{
	EXPECT_EQ(a.magnitude(), sqrt(2));
	EXPECT_EQ(b.magnitude(), sqrt(3));
}

TEST_F(Vector3dTest, DotProduct)
{
	EXPECT_EQ(a*b, 2);
}

} // namespace

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
