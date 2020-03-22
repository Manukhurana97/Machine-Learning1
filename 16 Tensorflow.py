import tensorflow as tf

X1 = tf.constant([1, 2, 3, 4])
x2 = tf.constant([5, 6, 7, 8])


result = tf.multiply(X1, x2)


print(result)

# creates a new context in which we can run the operation
sess = tf.compat.v1.Session()
print(sess.run(result))


#   (Placeholder)

x3 = tf.placeholder(tf.float32)
x4 = tf.placeholder(tf.float32)

result1 = tf.add(x3, x4)

sess1 = tf.Session()

print(sess.run(result1, {x3: [1, 2], x4: [4, 5]}))


#   (placeholder - multidimention)

x5 = tf.placeholder(tf.float32, shape=(2, 1))
x6 = tf.placeholder(tf.float32, shape=(1, 2))

result2 = tf.matmul(x5, x6)

sess2 =tf.compat.v1.Session()
print(sess2.run(result2, {x5: [[1], [2]], x6: [[3, 4]]}))

sess.close()
sess1.close()
sess2.close()
