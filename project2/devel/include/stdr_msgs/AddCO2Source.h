// Generated by gencpp from file stdr_msgs/AddCO2Source.msg
// DO NOT EDIT!


#ifndef STDR_MSGS_MESSAGE_ADDCO2SOURCE_H
#define STDR_MSGS_MESSAGE_ADDCO2SOURCE_H

#include <ros/service_traits.h>


#include <stdr_msgs/AddCO2SourceRequest.h>
#include <stdr_msgs/AddCO2SourceResponse.h>


namespace stdr_msgs
{

struct AddCO2Source
{

typedef AddCO2SourceRequest Request;
typedef AddCO2SourceResponse Response;
Request request;
Response response;

typedef Request RequestType;
typedef Response ResponseType;

}; // struct AddCO2Source
} // namespace stdr_msgs


namespace ros
{
namespace service_traits
{


template<>
struct MD5Sum< ::stdr_msgs::AddCO2Source > {
  static const char* value()
  {
    return "0dabebb840d5db7f089c1859d37b6027";
  }

  static const char* value(const ::stdr_msgs::AddCO2Source&) { return value(); }
};

template<>
struct DataType< ::stdr_msgs::AddCO2Source > {
  static const char* value()
  {
    return "stdr_msgs/AddCO2Source";
  }

  static const char* value(const ::stdr_msgs::AddCO2Source&) { return value(); }
};


// service_traits::MD5Sum< ::stdr_msgs::AddCO2SourceRequest> should match
// service_traits::MD5Sum< ::stdr_msgs::AddCO2Source >
template<>
struct MD5Sum< ::stdr_msgs::AddCO2SourceRequest>
{
  static const char* value()
  {
    return MD5Sum< ::stdr_msgs::AddCO2Source >::value();
  }
  static const char* value(const ::stdr_msgs::AddCO2SourceRequest&)
  {
    return value();
  }
};

// service_traits::DataType< ::stdr_msgs::AddCO2SourceRequest> should match
// service_traits::DataType< ::stdr_msgs::AddCO2Source >
template<>
struct DataType< ::stdr_msgs::AddCO2SourceRequest>
{
  static const char* value()
  {
    return DataType< ::stdr_msgs::AddCO2Source >::value();
  }
  static const char* value(const ::stdr_msgs::AddCO2SourceRequest&)
  {
    return value();
  }
};

// service_traits::MD5Sum< ::stdr_msgs::AddCO2SourceResponse> should match
// service_traits::MD5Sum< ::stdr_msgs::AddCO2Source >
template<>
struct MD5Sum< ::stdr_msgs::AddCO2SourceResponse>
{
  static const char* value()
  {
    return MD5Sum< ::stdr_msgs::AddCO2Source >::value();
  }
  static const char* value(const ::stdr_msgs::AddCO2SourceResponse&)
  {
    return value();
  }
};

// service_traits::DataType< ::stdr_msgs::AddCO2SourceResponse> should match
// service_traits::DataType< ::stdr_msgs::AddCO2Source >
template<>
struct DataType< ::stdr_msgs::AddCO2SourceResponse>
{
  static const char* value()
  {
    return DataType< ::stdr_msgs::AddCO2Source >::value();
  }
  static const char* value(const ::stdr_msgs::AddCO2SourceResponse&)
  {
    return value();
  }
};

} // namespace service_traits
} // namespace ros

#endif // STDR_MSGS_MESSAGE_ADDCO2SOURCE_H
