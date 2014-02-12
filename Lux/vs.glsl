/////////////////////////////// Vertex shader
#version 120

attribute vec4 iVS_Position;
attribute vec4 iVS_Normal;
attribute vec2 iVS_UV;
attribute vec4 iVS_Tangent;
attribute vec4 iVS_Binormal;

varying vec3 iFS_Normal;
varying vec2 iFS_UV;
varying vec3 iFS_Tangent;
varying vec3 iFS_Binormal;
varying vec3 iFS_PointWS;

uniform mat4 worldMatrix;
uniform mat4 worldViewProjMatrix;
uniform mat4 worldInverseTransposeMatrix;

void main()
{
    gl_Position = worldViewProjMatrix * iVS_Position;
    iFS_Normal = (worldInverseTransposeMatrix * iVS_Normal).xyz;
    iFS_UV = iVS_UV;
    iFS_Tangent = (worldInverseTransposeMatrix * iVS_Tangent).xyz;
    iFS_Binormal = (worldInverseTransposeMatrix * iVS_Binormal).xyz;
    iFS_PointWS = (worldMatrix * iVS_Position).xyz;
}
