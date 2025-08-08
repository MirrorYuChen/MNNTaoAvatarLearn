 /*
 * @Author: chenjingyu
 * @Date: 2025-08-08 19:05:45
 * @Contact: 2458006466@qq.com
 * @Description: AnToCn
 */
#pragma once

#include "Api.h"
#include "Utils/Utils.h"

NAMESPACE_BEGIN
class An2Cn {
public:
  // 类初始化函数
  An2Cn();

  // 处理主入口
  std::string Process(const std::string &inputs);

private:
  // 根据小数点对字符串进行切分
  std::vector<std::string> SplitByDot(const std::string &str);

  // 检查输入是否是合法的阿拉伯数字（只包含阿拉伯0-9数字和小数点和负号)
  bool CheckInputsIsValid(const std::string &inputs);

  // 处理整数部分
  std::string IntegerConvert(const std::string &inputs);

  // 处理小数部分
  std::string DecimalConvert(const std::string &inputs);

  // 去掉开头的0
  std::string RemovePrefixZero(const std::string &str);

private:
  std::string valid_chars_ = "0123456789.-";
  std::map<int, std::string> NUMBER_LOW_AN2CN = {
      {0, "零"}, {1, "一"}, {2, "二"}, {3, "三"}, {4, "四"},
      {5, "五"}, {6, "六"}, {7, "七"}, {8, "八"}, {9, "九"}};

  std::vector<std::string> UNIT_LOW_ORDER_AN2CN = {
      "",   "十", "百", "千", "万", "十", "百", "千",
      "亿", "十", "百", "千", "万", "十", "百", "千"};
};
NAMESPACE_END
