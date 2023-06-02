#include <gtest/gtest.h>

#include "../Headers/ringbuf.hpp"

typedef RingBuffer<int> IntRingBuffer;

TEST(Ringbuf, simplest)
{
}

TEST(Ringbuf, emptyBuf1)
{
    IntRingBuffer buf(5);

    EXPECT_EQ(5, buf.getSize());
    EXPECT_EQ(0, buf.getCount());
    EXPECT_TRUE(buf.isEmpty());
    EXPECT_FALSE(buf.isFull());
}

TEST(Ringbuf, push1)
{
    IntRingBuffer buf(3);

    EXPECT_EQ(3, buf.getSize());
    EXPECT_EQ(0, buf.getCount());
    EXPECT_TRUE(buf.isEmpty());
    EXPECT_FALSE(buf.isFull());
    EXPECT_EQ(3, buf.getFree());

    buf.push(1);
    EXPECT_EQ(1, buf.getCount());
    EXPECT_FALSE(buf.isEmpty());
    EXPECT_FALSE(buf.isFull());
    EXPECT_EQ(2, buf.getFree());

    buf.push(2);
    EXPECT_EQ(2, buf.getCount());
    EXPECT_FALSE(buf.isEmpty());
    EXPECT_FALSE(buf.isFull());
    EXPECT_EQ(1, buf.getFree());

    buf.push(3);
    EXPECT_EQ(3, buf.getCount());
    EXPECT_FALSE(buf.isEmpty());
    EXPECT_TRUE(buf.isFull());
    EXPECT_EQ(0, buf.getFree());

    EXPECT_THROW(
                buf.push(4),
                std::out_of_range);
}

TEST(Ringbuf, pushBackFront1)
{
    IntRingBuffer buf(3);

    EXPECT_THROW(
                buf.front(),
                std::out_of_range);

    EXPECT_THROW(
                buf.back(),
                std::out_of_range);

    buf.push(10);
    EXPECT_EQ(10, buf.back());
    EXPECT_EQ(10, buf.front());

    buf.push(20);
    EXPECT_EQ(20, buf.back());
    EXPECT_EQ(10, buf.front());

    buf.push(30);
    EXPECT_EQ(30, buf.back());
    EXPECT_EQ(10, buf.front());
}

TEST(Ringbuf, pop1)
{
    IntRingBuffer buf(3);
    EXPECT_THROW(
                buf.pop(),
                std::out_of_range);

    EXPECT_EQ(3, buf.getFree());

    buf.push(10);
    EXPECT_EQ(10, buf.back());
    EXPECT_EQ(10, buf.front());
    EXPECT_EQ(2, buf.getFree());

    buf.push(20);
    EXPECT_EQ(20, buf.back());
    EXPECT_EQ(10, buf.front());
    EXPECT_EQ(2, buf.getCount());
    EXPECT_EQ(1, buf.getFree());

    buf.pop();
    EXPECT_EQ(20, buf.back());
    EXPECT_EQ(20, buf.front());
    EXPECT_EQ(1, buf.getCount());
    EXPECT_EQ(2, buf.getFree());

    buf.pop();
    EXPECT_EQ(0, buf.getCount());
    EXPECT_TRUE(buf.isEmpty());
    EXPECT_EQ(3, buf.getFree());
}
