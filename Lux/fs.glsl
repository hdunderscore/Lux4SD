//////////////////////////////// Fragment shader
#version 120

varying vec3 iFS_Normal;
varying vec2 iFS_UV;
varying vec3 iFS_Tangent;
varying vec3 iFS_Binormal;
varying vec3 iFS_PointWS;

uniform vec3 Lamp0Pos = vec3(0.0f,0.0f,70.0f);
uniform vec3 Lamp0Color = vec3(1.0f,1.0f,1.0f);
uniform float Lamp0Intensity = 1.0f;
uniform vec3 Lamp1Pos = vec3(70.0f,0.0f,0.0f);
uniform vec3 Lamp1Color = vec3(0.198f,0.198f,0.198f);
uniform float Lamp1Intensity = 1.0f;

uniform float Kr = 0.5f;
uniform vec3 ambientColor = vec3(0.07f,0.07f,0.07f);
uniform float tiling = 1.0f;

uniform bool flipY = true;
uniform float heightMapScale = 3.0f;

uniform bool LUX_AO = true;
uniform bool LUX_AO_TILE = false;
uniform bool LUX_LINEAR = false;
uniform float HDR_Scale = 6.0f;
uniform bool DiffHDR = false;
uniform bool SpecHDR = false;
uniform float DiffuseExposure=1.0f;
uniform float SpecularExposure=1.0f;
uniform float DiffuseLOD = 5.0f;
uniform float Gamma = 2.22f;

uniform bool Gamma_Pow = true;
uniform bool Gamma_True = false;
uniform bool Gamma_Fast = false;

uniform float Lamp0Power=2.0f;
uniform float Lamp1Power=1.0f;

uniform sampler2D normalMap;
uniform sampler2D diffuseMap;
uniform sampler2D specularMap;
uniform sampler2D emissiveMap;
uniform samplerCube environmentMap;
uniform sampler2D ambientOcclusionMap;

uniform mat4 viewInverseMatrix;

struct SurfaceOutputLux {
	vec3 Albedo;
	vec3 Normal;
	vec3 Emission;
	float Specular;
	vec3 SpecularColor;
	float Alpha;
	float DeferredFresnel;
};

vec3 fixNormalSample(vec3 v)
{
  vec3 result = v - vec3(0.5f,0.5f,0.5f);
  result.y = flipY ? -result.y : result.y;
  return result;
}

vec3 normalVecOSToWS(vec3 normal)
{
  return normal;
}

#define OneOnLN2_x6 8.656170f
#define M_PI 3.1415926535897932384626433832795f

float CalcAtten(vec3 P, vec3 N, vec3 lightCentre, float lightRadius, float cutoff)
{
	//source: http://imdoingitwrong.wordpress.com/tag/glsl/
    // calculate normalized light vector and distance to sphere light surface
    float r = lightRadius;
    vec3 L = lightCentre - P;
    float distance = length(L);
    float d = max(distance - r, 0.0f);
    L /= distance;
     
    // calculate basic attenuation
    float denom = d/r + 1.0f;
    float attenuation = 1.0f / (denom*denom);
     
    // scale and bias attenuation such that:
    //   attenuation == 0 at extent of max influence
    //   attenuation == 1 when d == 0
    attenuation = (attenuation - cutoff) / (1.0f - cutoff);
    attenuation = max(attenuation, 0.0f);
    
    return attenuation;
}

vec4 LightingLuxPhongNDF (SurfaceOutputLux s, vec4 _LightColor0, vec3 lightDir, vec3 viewDir, float atten){
  
	//Lux License

	//Copyright (c) <2014> <larsbertam69@googlemail.com>
	//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
	//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
	//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  	
  	// get base variables

  	// normalizing lightDir makes fresnel smoother
	lightDir = normalize(lightDir);

	vec3 h = normalize (lightDir + viewDir);
	// dotNL has to have max
	float dotNL = max (0.0f, dot (s.Normal, lightDir));
	float dotNH = max (0.0f, dot (s.Normal, h));
	
	// bring specPower into a range of 0.25 – 2048
	float specPower = exp2(10.0f * s.Specular + 1) - 1.75f;

	// Normalized Lighting Model:
	// L = (c_diff * dotNL + F_Schlick(c_spec, l_c, h) * ( (spec + 2.0f)/8.0f) * dotNH˄spec * dotNL) * c_light
	
	// Specular: Phong lobe normal distribution function
	//float spec = ((specPower + 2.0) * 0.125 ) * pow(dotNH, specPower) * dotNL; // would be the correct term
	// we use late * dotNL to get rid of any artifacts on the backsides
	float spec = specPower * 0.125f * pow(dotNH, specPower);

	// Visibility: Schlick-Smith
	float alpha = 2.0f / sqrt( 3.14159265f * (specPower + 2.0f) );
	float visibility = 1.0f / ( (dotNL * (1.0f - alpha) + alpha) * ( dot(s.Normal, viewDir) * (1.0f - alpha) + alpha) ); 
	spec *= visibility;
	
	// Fresnel: Schlick
	// fixed3 fresnel = s.SpecularColor.rgb + ( 1.0f - s.SpecularColor.rgb) * pow(1.0f - saturate(dot(h, lightDir)), 5.0f);
	// fast fresnel approximation:
	vec3 fresnel = s.SpecularColor.rgb + ( 1.0f - s.SpecularColor.rgb) * exp2(-OneOnLN2_x6 * dot(h, lightDir));
	// using dot(h, lightDir) or dot(h, viewDir) does not make any difference
	//fixed3 fresnel = s.SpecularColor.rgb + ( 1.0f - s.SpecularColor.rgb) * exp2(-OneOnLN2_x6 * dot(h, viewDir));

	// from here on we use fresnel instead of spec as it is fixed3 = color
	fresnel *= spec;
	
	

	// Final Composition
	vec4 c;
	// we only use fresnel here / and apply late dotNL
	c.rgb = (s.Albedo.rgb * _LightColor0.rgb + _LightColor0.rgb * fresnel) * dotNL * (atten * 2.0f);
		
	c.a = s.Alpha + _LightColor0.a * atten * (fresnel.r + fresnel.g + fresnel.b);
	return c;
}

float saturate(float a)
{
	return max(0.0f,min(a,1.0f));
}

vec4 toLinearPow(vec4 col)
{
	vec4 ncol;
	
	ncol.r = pow(col.r,Gamma);
	ncol.g = pow(col.g,Gamma);
	ncol.b = pow(col.b,Gamma);
	ncol.a = pow(col.a,Gamma);
	
	return ncol;
}

vec4 toLinearTrue(vec4 col)
{
	vec4 ncol;

	if (col.r <= 0.04045)
	{ncol.r = col.r / 12.92;}
	else
	{ncol.r = pow((col.r + 0.055) / 1.055, 2.4);}
	
	if (col.g <= 0.04045)
	{ncol.g = col.g / 12.92;}
	else
	{ncol.g = pow((col.g + 0.055) / 1.055, 2.4);}
	
	if (col.b <= 0.04045)
	{ncol.b = col.b / 12.92;}
	else
	{ncol.b = pow((col.b + 0.055) / 1.055, 2.4);}
	
	if (col.a <= 0.04045)
	{ncol.a = col.a / 12.92;}
	else
	{ncol.a = pow((col.a + 0.055) / 1.055, 2.4);}
	
	return ncol;
}

vec4 toLinearFast(vec4 col)
{
	//http://chilliant.blogspot.de/2012/08/srgb-approximations-for-hlsl.html
	vec4 ncol;
	
	ncol.r = col.r * (col.r * (col.r * 0.305306011 + 0.682171111) + 0.012522878);
	ncol.g = col.g * (col.g * (col.g * 0.305306011 + 0.682171111) + 0.012522878);
	ncol.b = col.b * (col.b * (col.b * 0.305306011 + 0.682171111) + 0.012522878);
	ncol.a = col.a * (col.a * (col.a * 0.305306011 + 0.682171111) + 0.012522878);
	
	return ncol;
}

vec4 toLinear(vec4 col)
{
	
	if (Gamma_Pow)
	{
		return toLinearPow(col);
	}
	else if (Gamma_True)
	{
		return toLinearTrue(col);
	}
	else if (Gamma_Fast)
	{
		return toLinearFast(col);
	}
	else
	{
		return toLinearPow(col);
	}
}

vec4 toGammaPow(vec4 col)
{
	vec4 ncol;
	
	ncol.r = pow(col.r,1/Gamma);
	ncol.g = pow(col.g,1/Gamma);
	ncol.b = pow(col.b,1/Gamma);
	ncol.a = pow(col.a,1/Gamma);
	
	return ncol;
}

vec4 toGammaTrue(vec4 col)
{
	vec4 ncol;
	
	if (col.r <= 0.0031308)
	{ncol.r = col.r * 12.92;}
	else
	{ncol.r = 1.055 * pow(col.r, 1.0 / 2.4) - 0.055;}
	
	if (col.g <= 0.0031308)
	{ncol.g = col.g * 12.92;}
	else
	{ncol.g = 1.055 * pow(col.g, 1.0 / 2.4) - 0.055;}
	
	if (col.b <= 0.0031308)
	{ncol.b = col.b * 12.92;}
	else
	{ncol.b = 1.055 * pow(col.b, 1.0 / 2.4) - 0.055;}
	
	if (col.a <= 0.0031308)
	{ncol.a = col.a * 12.92;}
	else
	{ncol.a = 1.055 * pow(col.a, 1.0 / 2.4) - 0.055;}
	
	return ncol;
}

vec4 toGammaFast(vec4 col)
{
	//http://chilliant.blogspot.de/2012/08/srgb-approximations-for-hlsl.html
	
	vec4 ncol;
	
	vec3 S1 = sqrt(col.rgb);
 	vec3 S2 = sqrt(S1);
	vec3 S3 = sqrt(S2);
	ncol.rgb = 0.662002687 * S1 + 0.684122060 * S2 - 0.323583601 * S3 - 0.225411470 * col.rgb;
	
	ncol.a=1.0f; // nasty hack: calc brings in false alpha.
	
	return ncol;
}



vec4 toGamma(vec4 col)
{
	if (Gamma_Pow)
	{
		return toGammaPow(col);
	}
	else if (Gamma_True)
	{
		return toGammaTrue(col);
	}
	else if (Gamma_Fast)
	{
		return toGammaFast(col);
	}
	else
	{
		return toGammaPow(col);
	}
}

void main()
{
	vec3 cameraPosWS = viewInverseMatrix[3].xyz;
	vec3 pointToLight0DirWS = normalize(Lamp0Pos - iFS_PointWS);
	vec3 pointToLight1DirWS = normalize(Lamp1Pos - iFS_PointWS);
	vec3 pointToCameraDirWS = normalize(cameraPosWS - iFS_PointWS);
	vec3 normalOS = iFS_Normal;
	vec3 tangentOS = iFS_Tangent;
	vec3 binormalOS = iFS_Binormal;

	// ------------------------------------------
	vec3 cumulatedNormalOS = normalOS;

	// ------------------------------------------
	// Update UV
	vec2 uv = iFS_UV * tiling;

	// ------------------------------------------
	// Add Normal from normalMap
	vec3 normalTS = texture2D(normalMap,uv).xyz;
	if (length(normalTS)<0.0001f)
	{
		cumulatedNormalOS = normalOS;
	}
	else
	{
		normalTS = fixNormalSample(normalTS);
	    normalTS *= heightMapScale;
		vec3 normalMapOS = normalTS.x*tangentOS + normalTS.y*binormalOS;
		cumulatedNormalOS = cumulatedNormalOS + normalMapOS;
		cumulatedNormalOS = normalize(cumulatedNormalOS);
	}


  	vec3 cumulatedNormalWS = normalVecOSToWS(cumulatedNormalOS);

  	// ------------------------------------------
  	// Compute Diffuse & Specular

	SurfaceOutputLux o;
	
  	//float glossiness = texture2D(glossinessMap,uv).r * glossMult;
  
	vec4 diff_albedo = texture2D(diffuseMap, uv);
	vec4 spec_albedo = texture2D(specularMap, uv);
	
	if (LUX_LINEAR)
	{
		vec4 ldiff = toLinear(diff_albedo);
		vec4 lspec = toLinear(spec_albedo);
		spec_albedo.rgb = lspec.rgb;
		diff_albedo.rgb = ldiff.rgb;
	}
	
	// Diffuse Albedo
	o.Albedo = diff_albedo.rgb;// * _Color.rgb;
	o.Alpha = diff_albedo.a;// * _Color.a;
	o.Normal = cumulatedNormalWS; //UnpackNormal(tex2D(_BumpMap, uv));
	// Specular Color
	o.SpecularColor = spec_albedo.rgb;
	// Roughness – we just take it as it is and do not bring it into linear space (alpha!) so roughness textures must be authored using gamma values
	o.Specular = spec_albedo.a;
	

	// Light 0 contribution
	vec4 contrib1 = vec4(0.0f,0.0f,0.0f,0.0f);
	float atten1 = CalcAtten( iFS_PointWS, cumulatedNormalWS, Lamp0Pos, 8.0f, 20.0f);
	contrib1 = LightingLuxPhongNDF(o,vec4(Lamp0Color * Lamp0Intensity * Lamp0Power,1.0f), pointToLight0DirWS, pointToCameraDirWS,atten1);//, diffContrib, specContrib,atten1);

	// Light 1 contribution
	vec4 contrib2 = vec4(0.0f,0.0f,0.0f,0.0f);
	float atten2 = CalcAtten( iFS_PointWS, cumulatedNormalWS, Lamp1Pos, 8.0f, 20.0f);
	contrib2 = LightingLuxPhongNDF(o,vec4(Lamp1Color * Lamp1Intensity * Lamp1Power,1.0f), pointToLight1DirWS, pointToCameraDirWS,atten2);//, diffContrib1, specContrib1,atten2);

	// Lux IBL / ambient lighting
	float Lux_HDR_Scale = HDR_Scale;
	vec4 ExposureIBL;
	ExposureIBL.x=DiffuseExposure;
	ExposureIBL.y=SpecularExposure;

	if (DiffHDR)
	{
		ExposureIBL.x *= pow(Lux_HDR_Scale,2.2333333f);
	}
	if (SpecHDR)
	{
		ExposureIBL.y *= pow(Lux_HDR_Scale,2.2333333f);
	}


	//	vec3 worldNormal = WorldNormalVector (IN, o.Normal);	
	vec3 worldNormal = vec3(o.Normal.x,-o.Normal.y,o.Normal.z) ;
	
	//		add diffuse IBL
	vec4	diff_ibl = textureCubeLod (environmentMap, worldNormal, DiffuseLOD);
	diff_ibl = diff_ibl.bgra;
	
	if (LUX_LINEAR){
			// if colorspace = linear alpha has to be brought to linear too (rgb already is): alpha = pow(alpha,2.233333333).
			// approximation taken from http://chilliant.blogspot.de/2012/08/srgb-approximations-for-hlsl.html
			diff_ibl.a *= diff_ibl.a * (diff_ibl.a * 0.305306011f + 0.682171111f) + 0.012522878f;
	}
	diff_ibl.rgb = diff_ibl.rgb * diff_ibl.a;
	o.Emission = diff_ibl.rgb * ExposureIBL.x * o.Albedo;

	//		add specular IBL		
	//vec3 worldRefl = WorldReflectionVector (IN, o.Normal);
	vec3 worldRefl = reflect(pointToCameraDirWS,cumulatedNormalWS);
	vec4 spec_ibl = textureCubeLod (environmentMap, worldRefl, 6.0f - 8.0f * spec_albedo.a);
	spec_ibl = spec_ibl.bgra;

	if (LUX_LINEAR){
		// if colorspace = linear alpha has to be brought to linear too (rgb already is): alpha = pow(alpha,2.233333333f).
		// approximation taken from http://chilliant.blogspot.de/2012/08/srgb-approximations-for-hlsl.html
		spec_ibl.a *= spec_ibl.a * (spec_ibl.a * 0.305306011f + 0.682171111f) + 0.012522878f;
	}
	
	spec_ibl.rgb = spec_ibl.rgb * spec_ibl.a;
	// fresnel based on spec_albedo.rgb and roughness (spec_albedo.a)
	// taken from: http://seblagarde.wordpress.com/2011/08/17/hello-world/
	// viewDir is in tangent-space (as we sample o.Normal) so we use o.Normal
	vec3 FresnelSchlickWithRoughness = spec_albedo.rgb + ( max(spec_albedo.aaa,spec_albedo.rgb) - spec_albedo.rgb) * exp2(-OneOnLN2_x6 * saturate(dot(normalize(pointToCameraDirWS), o.Normal)));		
	spec_ibl.rgb *= FresnelSchlickWithRoughness * ExposureIBL.y;
	
	// add diffuse and specular and conserve energy
	o.Emission = (1.0f - spec_ibl.rgb) * o.Emission + spec_ibl.rgb;
	
	vec4 Ambient_final = vec4(o.Albedo.rgb * ambientColor.rgb,1.0f);
	if (LUX_AO){
		vec2 AO_uv = uv;
		if (!LUX_AO_TILE)
		{
			AO_uv = iFS_UV;
		}
		vec4 ambientOcclusion = texture2D(ambientOcclusionMap, AO_uv);
		o.Emission *= ambientOcclusion.a;
		Ambient_final.a = ambientOcclusion.a;
	}
   
   
   	vec4 emis_albedo = texture2D(emissiveMap, uv);
	o.Emission += emis_albedo.rgb * emis_albedo.a;
   
  	// ------------------------------------------
  	vec3 finalcolor = contrib1.rgb * contrib1.a + contrib2.rgb * contrib2.a + (o.Emission + Ambient_final.rgb * Ambient_final.a);

 	// Final Color
 	if (LUX_LINEAR)
 	{
  		gl_FragColor = toGamma(vec4(finalcolor,1.0f));
  	}
  	else
  	{
  		gl_FragColor = vec4(finalcolor,1.0f);
  	}
}

