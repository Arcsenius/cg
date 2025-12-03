#pragma once

#include <cstdint>
#include <cmath>
#include <limits>

namespace veekay {

union vec2 {
	struct {
		float x;
		float y;
	};

	float elements[2];

	vec2& operator+=(const vec2& other) {
		x += other.x;
		y += other.y;
		return *this;
	}

	vec2& operator+=(float scalar) {
		x += scalar;
		y += scalar;
		return *this;
	}

	vec2& operator-=(const vec2& other) {
		x -= other.x;
		y -= other.y;
		return *this;
	}

	vec2& operator-=(float scalar) {
		x -= scalar;
		y -= scalar;
		return *this;
	}

	vec2& operator*=(const vec2& other) {
		x *= other.x;
		y *= other.y;
		return *this;
	}

	vec2& operator*=(float scalar) {
		x *= scalar;
		y *= scalar;
		return *this;
	}

	vec2& operator/=(const vec2& other) {
		x /= other.x;
		y /= other.y;
		return *this;
	}

	vec2& operator/=(float scalar) {
		x /= scalar;
		y /= scalar;
		return *this;
	}

	vec2 operator+(const vec2& other) const {
		vec2 result = *this;
		return result += other;
	}

	vec2 operator+(float scalar) const {
		vec2 result = *this;
		return result += scalar;
	}

	vec2 operator-(const vec2& other) const {
		vec2 result = *this;
		return result -= other;
	}

	vec2 operator-(float scalar) const {
		vec2 result = *this;
		return result -= scalar;
	}

	vec2 operator-() const {
		return {-x, -y};
	}

	vec2 operator*(const vec2& other) const {
		vec2 result = *this;
		return result *= other;
	}

	vec2 operator*(float scalar) const {
		vec2 result = *this;
		return result *= scalar;
	}

	vec2 operator/(const vec2& other) const {
		vec2 result = *this;
		return result /= other;
	}

	vec2 operator/(float scalar) const {
		vec2 result = *this;
		return result /= scalar;
	}

	float& operator[](size_t index) { return elements[index]; }
	const float& operator[](size_t index) const { return elements[index]; }
};

union vec3 {
	struct {
		float x;
		float y;
		float z;
	};

	float elements[3];

	vec3& operator+=(const vec3& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	vec3& operator+=(float scalar) {
		x += scalar;
		y += scalar;
		z += scalar;
		return *this;
	}

	vec3& operator-=(const vec3& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		return *this;
	}

	vec3& operator-=(float scalar) {
		x -= scalar;
		y -= scalar;
		z -= scalar;
		return *this;
	}

	vec3& operator*=(const vec3& other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		return *this;
	}

	vec3& operator*=(float scalar) {
		x *= scalar;
		y *= scalar;
		z *= scalar;
		return *this;
	}

	vec3& operator/=(const vec3& other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		return *this;
	}

	vec3& operator/=(float scalar) {
		x /= scalar;
		y /= scalar;
		z /= scalar;
		return *this;
	}

	vec3 operator+(const vec3& other) const {
		vec3 result = *this;
		return result += other;
	}

	vec3 operator+(float scalar) const {
		vec3 result = *this;
		return result += scalar;
	}

	vec3 operator-(const vec3& other) const {
		vec3 result = *this;
		return result -= other;
	}

	vec3 operator-(float scalar) const {
		vec3 result = *this;
		return result -= scalar;
	}

	vec3 operator-() const { return {-x, -y, -z}; }

	vec3 operator*(const vec3& other) const {
		vec3 result = *this;
		return result *= other;
	}

	vec3 operator*(float scalar) const {
		vec3 result = *this;
		return result *= scalar;
	}

	vec3 operator/(const vec3& other) const {
		vec3 result = *this;
		return result /= other;
	}

	vec3 operator/(float scalar) const {
		vec3 result = *this;
		return result /= scalar;
	}

	static float dot(const vec3& lhs, const vec3& rhs) {
		return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
	}

	static float squaredLength(const vec3& vector) {
		return dot(vector, vector);
	}

	static float length(const vec3& vector) {
		return std::sqrt(squaredLength(vector));
	}

	static vec3 normalized(const vec3& vector) {
		vec3 result = vector;

		float len = length(vector);

		result.x /= len;
		result.y /= len;
		result.z /= len;

		return result;
	}

	static vec3 cross(const vec3& lhs, const vec3& rhs) {
		return {
			(lhs.y * rhs.z) - (lhs.z * rhs.y),
			(lhs.z * rhs.x) - (lhs.x * rhs.z),
			(lhs.x * rhs.y) - (lhs.y * rhs.x)
		};
	}

	float& operator[](size_t index) { return elements[index]; }
	const float& operator[](size_t index) const { return elements[index]; }
};

union vec4 {
	struct {
		float x;
		float y;
		float z;
		float w;
	};

	float elements[4];

	vec4& operator+=(const vec4& other) {
		x += other.x;
		y += other.y;
		z += other.z;
		w += other.w;
		return *this;
	}

	vec4& operator-=(const vec4& other) {
		x -= other.x;
		y -= other.y;
		z -= other.z;
		w -= other.w;
		return *this;
	}

	vec4& operator*=(const vec4& other) {
		x *= other.x;
		y *= other.y;
		z *= other.z;
		w *= other.w;
		return *this;
	}

	vec4& operator/=(const vec4& other) {
		x /= other.x;
		y /= other.y;
		z /= other.z;
		w /= other.w;
		return *this;
	}

	vec4 operator+(const vec4& other) const {
		vec4 result = *this;
		return result += other;
	}

	vec4 operator-(const vec4& other) const {
		vec4 result = *this;
		return result -= other;
	}

	vec4 operator*(const vec4& other) const {
		vec4 result = *this;
		return result *= other;
	}

	vec4 operator/(const vec4& other) const {
		vec4 result = *this;
		return result /= other;
	}

	float& operator[](size_t index) { return elements[index]; }
	const float& operator[](size_t index) const { return elements[index]; }
};

union mat4 {
	float elements[4][4];
	vec4 columns[4];

	static mat4 identity() {
		mat4 result{};

		result[0][0] = 1.0f;
		result[1][1] = 1.0f;
		result[2][2] = 1.0f;
		result[3][3] = 1.0f;
		
		return result;
	}

	static mat4 translation(vec3 vector) {
		mat4 result = mat4::identity();

		result[3][0] = vector.x;
		result[3][1] = vector.y;
		result[3][2] = vector.z;

		return result;
	}

	static mat4 scaling(vec3 vector) {
		mat4 result{};

		result[0][0] = vector.x;
		result[1][1] = vector.y;
		result[2][2] = vector.z;
		result[3][3] = 1.0f;

		return result;
	}

	static mat4 rotation(vec3 axis, float angle) {
		mat4 result{};

		float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

		axis.x /= length;
		axis.y /= length;
		axis.z /= length;

		float sina = sinf(angle);
		float cosa = cosf(angle);
		float cosv = 1.0f - cosa;

		result[0][0] = (axis.x * axis.x * cosv) + cosa;
		result[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
		result[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

		result[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
		result[1][1] = (axis.y * axis.y * cosv) + cosa;
		result[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

		result[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
		result[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
		result[2][2] = (axis.z * axis.z * cosv) + cosa;

		result[3][3] = 1.0f;

		return result;
	}

	static mat4 projection(float fov, float aspect_ratio, float near, float far) {
		mat4 result{};

		const float radians = fov * M_PI / 180.0f;
		const float cot = 1.0f / tanf(radians / 2.0f);

		result[0][0] = cot / aspect_ratio;
		result[1][1] = cot;
		result[2][3] = 1.0f;

		result[2][2] = far / (far - near);
		result[3][2] = (-near * far) / (far - near);

		return result;
	}

	static mat4 transpose(const mat4& matrix) {
		mat4 result{};

		for (int j = 0; j < 4; ++j) {
			for (int i = 0; i < 4; ++i) {
				result[j][i] = matrix[i][j];
			}
		}

		return result;
	}

	static mat4 inverse(const mat4& matrix) {
		float m[16];
		for (int col = 0; col < 4; ++col) {
			for (int row = 0; row < 4; ++row) {
				m[row * 4 + col] = matrix[col][row];
			}
		}

		float inv[16];

		inv[0] = m[5]  * m[10] * m[15] -
		         m[5]  * m[11] * m[14] -
		         m[9]  * m[6]  * m[15] +
		         m[9]  * m[7]  * m[14] +
		         m[13] * m[6]  * m[11] -
		         m[13] * m[7]  * m[10];

		inv[4] = -m[4]  * m[10] * m[15] +
		          m[4]  * m[11] * m[14] +
		          m[8]  * m[6]  * m[15] -
		          m[8]  * m[7]  * m[14] -
		          m[12] * m[6]  * m[11] +
		          m[12] * m[7]  * m[10];

		inv[8] = m[4]  * m[9] * m[15] -
		         m[4]  * m[11] * m[13] -
		         m[8]  * m[5] * m[15] +
		         m[8]  * m[7] * m[13] +
		         m[12] * m[5] * m[11] -
		         m[12] * m[7] * m[9];

		inv[12] = -m[4]  * m[9] * m[14] +
		           m[4]  * m[10] * m[13] +
		           m[8]  * m[5] * m[14] -
		           m[8]  * m[6] * m[13] -
		           m[12] * m[5] * m[10] +
		           m[12] * m[6] * m[9];

		inv[1] = -m[1]  * m[10] * m[15] +
		          m[1]  * m[11] * m[14] +
		          m[9]  * m[2] * m[15] -
		          m[9]  * m[3] * m[14] -
		          m[13] * m[2] * m[11] +
		          m[13] * m[3] * m[10];

		inv[5] = m[0]  * m[10] * m[15] -
		         m[0]  * m[11] * m[14] -
		         m[8]  * m[2] * m[15] +
		         m[8]  * m[3] * m[14] +
		         m[12] * m[2] * m[11] -
		         m[12] * m[3] * m[10];

		inv[9] = -m[0]  * m[9] * m[15] +
		          m[0]  * m[11] * m[13] +
		          m[8]  * m[1] * m[15] -
		          m[8]  * m[3] * m[13] -
		          m[12] * m[1] * m[11] +
		          m[12] * m[3] * m[9];

		inv[13] = m[0]  * m[9] * m[14] -
		          m[0]  * m[10] * m[13] -
		          m[8]  * m[1] * m[14] +
		          m[8]  * m[2] * m[13] +
		          m[12] * m[1] * m[10] -
		          m[12] * m[2] * m[9];

		inv[2] = m[1]  * m[6] * m[15] -
		         m[1]  * m[7] * m[14] -
		         m[5]  * m[2] * m[15] +
		         m[5]  * m[3] * m[14] +
		         m[13] * m[2] * m[7] -
		         m[13] * m[3] * m[6];

		inv[6] = -m[0]  * m[6] * m[15] +
		          m[0]  * m[7] * m[14] +
		          m[4]  * m[2] * m[15] -
		          m[4]  * m[3] * m[14] -
		          m[12] * m[2] * m[7] +
		          m[12] * m[3] * m[6];

		inv[10] = m[0]  * m[5] * m[15] -
		          m[0]  * m[7] * m[13] -
		          m[4]  * m[1] * m[15] +
		          m[4]  * m[3] * m[13] +
		          m[12] * m[1] * m[7] -
		          m[12] * m[3] * m[5];

		inv[14] = -m[0]  * m[5] * m[14] +
		           m[0]  * m[6] * m[13] +
		           m[4]  * m[1] * m[14] -
		           m[4]  * m[2] * m[13] -
		           m[12] * m[1] * m[6] +
		           m[12] * m[2] * m[5];

		inv[3] = -m[1] * m[6] * m[11] +
		          m[1] * m[7] * m[10] +
		          m[5] * m[2] * m[11] -
		          m[5] * m[3] * m[10] -
		          m[9] * m[2] * m[7] +
		          m[9] * m[3] * m[6];

		inv[7] = m[0] * m[6] * m[11] -
		         m[0] * m[7] * m[10] -
		         m[4] * m[2] * m[11] +
		         m[4] * m[3] * m[10] +
		         m[8] * m[2] * m[7] -
		         m[8] * m[3] * m[6];

		inv[11] = -m[0] * m[5] * m[11] +
		           m[0] * m[7] * m[9] +
		           m[4] * m[1] * m[11] -
		           m[4] * m[3] * m[9] -
		           m[8] * m[1] * m[7] +
		           m[8] * m[3] * m[5];

		inv[15] = m[0] * m[5] * m[10] -
		          m[0] * m[6] * m[9] -
		          m[4] * m[1] * m[10] +
		          m[4] * m[2] * m[9] +
		          m[8] * m[1] * m[6] -
		          m[8] * m[2] * m[5];

		float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
		if (std::fabs(det) <= std::numeric_limits<float>::epsilon()) {
			return mat4::identity();
		}

		det = 1.0f / det;
		mat4 result{};

		for (int i = 0; i < 16; ++i) {
			inv[i] *= det;
		}

		for (int col = 0; col < 4; ++col) {
			for (int row = 0; row < 4; ++row) {
				result[col][row] = inv[row * 4 + col];
			}
		}

		return result;
	}

	mat4 operator*(const mat4& other) const {
		mat4 result{};

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				for (int k = 0; k < 4; k++) {
					result[j][i] += elements[j][k] * other[k][i];
				}
			}
		}

		return result;
	}

	vec4& operator[](size_t index) { return columns[index]; }
	const vec4& operator[](size_t index) const { return columns[index]; }
};

} // namespace veekay
