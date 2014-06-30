#include "Vector3d.h"
#include "gtest/gtest.h"

TEST(Vector3dTest, Magnitude)
{
	EXPECT_EQ(Vector3d(0,1,1).magnitude(), sqrt(2));
	EXPECT_EQ(Vector3d(1,1,1).magnitude(), sqrt(3));
}

TEST(Vector3dTest, DotProduct)
{
	EXPECT_EQ(Vector3d(1,2,3)*Vector3d(4,5,6), 32);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
