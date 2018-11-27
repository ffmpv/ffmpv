/*
 * This file is part of mpv.
 * Parts based on MPlayer code by Reimar Döffinger.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <assert.h>

#include <libavutil/sha.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/mem.h>

#include "osdep/io.h"

#include "common/common.h"
#include "options/path.h"
#include "stream/stream.h"
#include "formats.h"
#include "utils.h"

#define CIMGUI_DEFINE_ENUMS_AND_STRUCTS 1
#include "cimgui.h"

static char         g_GlslVersionString[32] = "#version 440";
static GLuint       g_FontTexture = 0;
static GLuint       g_ShaderHandle = 0, g_VertHandle = 0, g_FragHandle = 0;
static int          g_AttribLocationTex = 0, g_AttribLocationProjMtx = 0;
static int          g_AttribLocationPosition = 0, g_AttribLocationUV = 0, g_AttribLocationColor = 0;
static unsigned int g_VboHandle = 0, g_ElementsHandle = 0;

static bool CheckShader(GL *gl, GLuint handle, const char* desc)
{
    GLint status = 0, log_length = 0;
    gl->GetShaderiv(handle, GL_COMPILE_STATUS, &status);
    gl->GetShaderiv(handle, GL_INFO_LOG_LENGTH, &log_length);
    if ((GLboolean)status == GL_FALSE)
        fprintf(stderr, "ERROR: failed to compile %s!\n", desc);
    if (log_length > 0) {
        uint8_t buf[log_length + 1];
        gl->GetShaderInfoLog(handle, log_length, NULL, (GLchar*)buf);
        fprintf(stderr, "%s\n", buf);
    }
    return (GLboolean)status == GL_TRUE;
}

static bool CheckProgram(GL *gl, GLuint handle, const char* desc)
{
    GLint status = 0, log_length = 0;
    gl->GetProgramiv(handle, GL_LINK_STATUS, &status);
    gl->GetProgramiv(handle, GL_INFO_LOG_LENGTH, &log_length);
    if ((GLboolean)status == GL_FALSE)
        fprintf(stderr, "ERROR: failed to link %s! (with GLSL '%s')\n", desc, g_GlslVersionString);
    return (GLboolean)status == GL_TRUE;
}

// GLU has this as gluErrorString (we don't use GLU, as it is legacy-OpenGL)
static const char *gl_error_to_string(GLenum error)
{
    switch (error) {
    case GL_INVALID_ENUM: return "INVALID_ENUM";
    case GL_INVALID_VALUE: return "INVALID_VALUE";
    case GL_INVALID_OPERATION: return "INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
    default: return "unknown";
    }
}

void gl_check_error(GL *gl, struct mp_log *log, const char *info)
{
    for (;;) {
        GLenum error = gl->GetError();
        if (error == GL_NO_ERROR)
            break;
        mp_msg(log, MSGL_ERR, "%s: OpenGL error %s.\n", info,
               gl_error_to_string(error));
    }
}

static int get_alignment(int stride)
{
    if (stride % 8 == 0)
        return 8;
    if (stride % 4 == 0)
        return 4;
    if (stride % 2 == 0)
        return 2;
    return 1;
}

// upload a texture, handling things like stride and slices
//  target: texture target, usually GL_TEXTURE_2D
//  format, type: texture parameters
//  dataptr, stride: image data
//  x, y, width, height: part of the image to upload
void gl_upload_tex(GL *gl, GLenum target, GLenum format, GLenum type,
                   const void *dataptr, int stride,
                   int x, int y, int w, int h)
{
    int bpp = gl_bytes_per_pixel(format, type);
    const uint8_t *data = dataptr;
    int y_max = y + h;
    if (w <= 0 || h <= 0 || !bpp)
        return;
    assert(stride > 0);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, get_alignment(stride));
    int slice = h;
    if (gl->mpgl_caps & MPGL_CAP_ROW_LENGTH) {
        // this is not always correct, but should work for MPlayer
        gl->PixelStorei(GL_UNPACK_ROW_LENGTH, stride / bpp);
    } else {
        if (stride != bpp * w)
            slice = 1; // very inefficient, but at least it works
    }
    for (; y + slice <= y_max; y += slice) {
        gl->TexSubImage2D(target, 0, x, y, w, slice, format, type, data);
        data += stride * slice;
    }
    if (y < y_max)
        gl->TexSubImage2D(target, 0, x, y, w, y_max - y, format, type, data);
    if (gl->mpgl_caps & MPGL_CAP_ROW_LENGTH)
        gl->PixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, 4);
}

bool gl_read_fbo_contents(GL *gl, int fbo, int dir, GLenum format, GLenum type,
                          int w, int h, uint8_t *dst, int dst_stride)
{
    assert(dir == 1 || dir == -1);
    if (fbo == 0 && gl->es)
        return false; // ES can't read from front buffer
    gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);
    GLenum obj = fbo ? GL_COLOR_ATTACHMENT0 : GL_FRONT;
    gl->PixelStorei(GL_PACK_ALIGNMENT, 1);
    gl->ReadBuffer(obj);
    // reading by line allows flipping, and avoids stride-related trouble
    int y1 = dir > 0 ? 0 : h;
    for (int y = 0; y < h; y++)
        gl->ReadPixels(0, y, w, 1, format, type, dst + (y1 + dir * y) * dst_stride);
    gl->PixelStorei(GL_PACK_ALIGNMENT, 4);
    gl->BindFramebuffer(GL_FRAMEBUFFER, 0);
    return true;
}

static void gl_vao_enable_attribs(struct gl_vao *vao)
{
    GL *gl = vao->gl;

    for (int n = 0; n < vao->num_entries; n++) {
        const struct ra_renderpass_input *e = &vao->entries[n];
        GLenum type = 0;
        bool normalized = false;
        switch (e->type) {
        case RA_VARTYPE_INT:
            type = GL_INT;
            break;
        case RA_VARTYPE_FLOAT:
            type = GL_FLOAT;
            break;
        case RA_VARTYPE_BYTE_UNORM:
            type = GL_UNSIGNED_BYTE;
            normalized = true;
            break;
        default:
            abort();
        }
        assert(e->dim_m == 1);

        gl->EnableVertexAttribArray(n);
        gl->VertexAttribPointer(n, e->dim_v, type, normalized,
                                vao->stride, (void *)(intptr_t)e->offset);
    }
}

static bool create_fonts_texture(struct gl_vao *vao)
{
    ImFontAtlas *font_atlas = ImFontAtlas_ImFontAtlas();
    GL *gl = vao->gl;
    ImGuiIO *io = igGetIO();
    unsigned char *pixels;
    int width, height, bpp;
    ImFontAtlas_GetTexDataAsRGBA32(font_atlas, &pixels, &width, &height, &bpp);

    // Upload texture to graphics system
    GLint last_texture;
    gl->GetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
    gl->GenTextures(1, &g_FontTexture);
    gl->BindTexture(GL_TEXTURE_2D, g_FontTexture);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl->PixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    gl->TexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    // Store our identifier
    io->Fonts->TexID = (ImTextureID)(intptr_t)g_FontTexture;

    // Restore state
    gl->BindTexture(GL_TEXTURE_2D, last_texture);

    return true;
}

static int gui_create(struct gl_vao *vao)
{
    GL *gl = vao->gl;

    // Backup GL state
    GLint last_texture, last_array_buffer, last_vertex_array;
    gl->GetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
    gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING, &last_array_buffer);
    gl->GetIntegerv(GL_VERTEX_ARRAY_BINDING, &last_vertex_array);

    // Parse GLSL version string
    int glsl_version = 130;
    sscanf(g_GlslVersionString, "#version %d", &glsl_version);

    const GLchar* vertex_shader_glsl_120 =
        "uniform mat4 ProjMtx;\n"
        "attribute vec2 Position;\n"
        "attribute vec2 UV;\n"
        "attribute vec4 Color;\n"
        "varying vec2 Frag_UV;\n"
        "varying vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "    Frag_UV = UV;\n"
        "    Frag_Color = Color;\n"
        "    gl_Position = ProjMtx * vec4(Position.xy,0,1);\n"
        "}\n";

    const GLchar* vertex_shader_glsl_130 =
        "uniform mat4 ProjMtx;\n"
        "in vec2 Position;\n"
        "in vec2 UV;\n"
        "in vec4 Color;\n"
        "out vec2 Frag_UV;\n"
        "out vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "Frag_UV = UV;\n"
        "Frag_Color = Color;\n"
        "gl_Position = ProjMtx * vec4(Position.xy,0,1);\n"
        "}\n";

    const GLchar* vertex_shader_glsl_300_es =
        "#version 300\n"
        "precision mediump float;\n"
        "layout(location=0) in vec2 Position;\n"
        "layout(location=1) in vec2 UV;\n"
        "layout(location=2) in vec4 Color;\n"
        "uniform mat4 ProjMtx;\n"
        "out vec2 Frag_UV;\n"
        "out vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "Frag_UV = UV;\n"
        "Frag_Color = Color;\n"
        "gl_Position = ProjMtx * vec4(Position.xy,0,1);\n"
        "}\n";

    const GLchar *vertex_shader_glsl_410_core =
        "#version 440\n"
        "layout(location=0) in vec2 Position;\n"
        "layout(location=1) in vec2 UV;\n"
        "layout(location=2) in vec4 Color;\n"
        "uniform mat4 ProjMtx;\n"
        "out vec2 Frag_UV;\n"
        "out vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "    Frag_UV = UV;\n"
        "    Frag_Color = Color;\n"
        "    gl_Position = ProjMtx * vec4(Position.xy, 0, 1);\n"
        "}\n";

    const GLchar* fragment_shader_glsl_120 =
        "#ifdef GL_ES\n"
        "    precision mediump float;\n"
        "#endif\n"
        "uniform sampler2D Texture;\n"
        "varying vec2 Frag_UV;\n"
        "varying vec4 Frag_Color;\n"
        "void main()\n"
        "{\n"
        "    gl_FragColor = Frag_Color * texture2D(Texture, Frag_UV.st);\n"
        "}\n";

    const GLchar* fragment_shader_glsl_130 =
        "uniform sampler2D Texture;\n"
        "in vec2 Frag_UV;\n"
        "in vec4 Frag_Color;\n"
        "out vec4 Out_Color;\n"
        "void main()\n"
        "{\n"
        "    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);\n"
        "}\n";

    const GLchar* fragment_shader_glsl_300_es =
        "#version 300\n"
        "precision mediump float;\n"
        "uniform sampler2D Texture;\n"
        "in vec2 Frag_UV;\n"
        "in vec4 Frag_Color;\n"
        "layout (location = 0) out vec4 Out_Color;\n"
        "void main()\n"
        "{\n"
        "    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);\n"
        "}\n";

    const GLchar* fragment_shader_glsl_410_core =
        "#version 440\n"
        "in vec2 Frag_UV;\n"
        "in vec4 Frag_Color;\n"
        "uniform sampler2D Texture;\n"
        "layout (location = 0) out vec4 Out_Color;\n"
        "void main()\n"
        "{\n"
        "    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);\n"
        "}\n";

    // Select shaders matching our GLSL versions
    const GLchar* vertex_shader = NULL;
    const GLchar* fragment_shader = NULL;
    if (glsl_version < 130)
    {
        vertex_shader = vertex_shader_glsl_120;
        fragment_shader = fragment_shader_glsl_120;
    }
    else if (glsl_version == 440)
    {
        vertex_shader = vertex_shader_glsl_410_core;
        fragment_shader = fragment_shader_glsl_410_core;
    }
    else if (glsl_version == 300)
    {
        vertex_shader = vertex_shader_glsl_300_es;
        fragment_shader = fragment_shader_glsl_300_es;
    }
    else
    {
        vertex_shader = vertex_shader_glsl_130;
        fragment_shader = fragment_shader_glsl_130;
    }

    // Create shaders
    const GLchar* vertex_shader_with_version[1] = { vertex_shader };
    g_VertHandle = gl->CreateShader(GL_VERTEX_SHADER);
    gl->ShaderSource(g_VertHandle, 1, vertex_shader_with_version, NULL);
    gl->CompileShader(g_VertHandle);
    CheckShader(gl, g_VertHandle, "vertex shader");

    const GLchar* fragment_shader_with_version[1] = { fragment_shader };
    g_FragHandle = gl->CreateShader(GL_FRAGMENT_SHADER);
    gl->ShaderSource(g_FragHandle, 1, fragment_shader_with_version, NULL);
    gl->CompileShader(g_FragHandle);
    CheckShader(gl, g_FragHandle, "fragment shader");

    g_ShaderHandle = gl->CreateProgram();
    gl->AttachShader(g_ShaderHandle, g_VertHandle);
    gl->AttachShader(g_ShaderHandle, g_FragHandle);
    gl->LinkProgram(g_ShaderHandle);
    CheckProgram(gl, g_ShaderHandle, "shader program");

    g_AttribLocationTex = gl->GetUniformLocation(g_ShaderHandle, "Texture");
    g_AttribLocationProjMtx = gl->GetUniformLocation(g_ShaderHandle, "ProjMtx");
    g_AttribLocationPosition = gl->GetAttribLocation(g_ShaderHandle, "Position");
    g_AttribLocationUV = gl->GetAttribLocation(g_ShaderHandle, "UV");
    g_AttribLocationColor = gl->GetAttribLocation(g_ShaderHandle, "Color");

    // Create buffers
    gl->GenBuffers(1, &g_VboHandle);
    gl->GenBuffers(1, &g_ElementsHandle);

    create_fonts_texture(vao);

    // Restore modified GL state
    gl->BindTexture(GL_TEXTURE_2D, last_texture);
    gl->BindBuffer(GL_ARRAY_BUFFER, last_array_buffer);
    gl->BindVertexArray(last_vertex_array);

    return true;
}

void gl_vao_init(struct gl_vao *vao, GL *gl, int stride,
                 const struct ra_renderpass_input *entries,
                 int num_entries)
{
    assert(!vao->vao);
    assert(!vao->buffer);

    *vao = (struct gl_vao){
        .gl = gl,
        .stride = stride,
        .entries = entries,
        .num_entries = num_entries,
    };

    gl->GenBuffers(1, &vao->buffer);

    if (gl->BindVertexArray) {
        gl->BindBuffer(GL_ARRAY_BUFFER, vao->buffer);

        gl->GenVertexArrays(1, &vao->vao);
        gl->BindVertexArray(vao->vao);
        gl_vao_enable_attribs(vao);
        gl->BindVertexArray(0);

        gl->BindBuffer(GL_ARRAY_BUFFER, 0);
    }

    gui_create(vao);
}

void gl_vao_uninit(struct gl_vao *vao)
{
    GL *gl = vao->gl;
    if (!gl)
        return;

    if (gl->DeleteVertexArrays)
        gl->DeleteVertexArrays(1, &vao->vao);
    gl->DeleteBuffers(1, &vao->buffer);

    *vao = (struct gl_vao){0};
}

static void gl_vao_bind(struct gl_vao *vao)
{
    GL *gl = vao->gl;

    if (gl->BindVertexArray) {
        gl->BindVertexArray(vao->vao);
    } else {
        gl->BindBuffer(GL_ARRAY_BUFFER, vao->buffer);
        gl_vao_enable_attribs(vao);
        gl->BindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

static void gl_vao_unbind(struct gl_vao *vao)
{
    GL *gl = vao->gl;

    if (gl->BindVertexArray) {
        gl->BindVertexArray(0);
    } else {
        for (int n = 0; n < vao->num_entries; n++)
            gl->DisableVertexAttribArray(n);
    }
}

static void gui_run(struct gl_vao *vao)
{
    GL *gl = vao->gl;
    ImGuiIO *io = igGetIO();
    ImDrawData *draw_data = igGetDrawData();
    ImVec2 pos;
    int fb_width;
    int fb_height;
    bool clip_origin_lower_left = true;

    if (!draw_data)
        return;

    fb_width = (int)(draw_data->DisplaySize.x * io->DisplayFramebufferScale.x);
    fb_height = (int)(draw_data->DisplaySize.y * io->DisplayFramebufferScale.y);
    if (fb_width <= 0 || fb_height <= 0)
        return;

    g_AttribLocationTex = gl->GetUniformLocation(g_ShaderHandle, "Texture");
    g_AttribLocationProjMtx = gl->GetUniformLocation(g_ShaderHandle, "ProjMtx");
    GLenum last_active_texture; gl->GetIntegerv(GL_ACTIVE_TEXTURE, (GLint*)&last_active_texture);
    gl->ActiveTexture(GL_TEXTURE0);
    GLint last_program; gl->GetIntegerv(GL_CURRENT_PROGRAM, &last_program);
    GLint last_texture; gl->GetIntegerv(GL_TEXTURE_BINDING_2D, &last_texture);
    GLint last_array_buffer; gl->GetIntegerv(GL_ARRAY_BUFFER_BINDING, &last_array_buffer);
    GLint last_vertex_array; gl->GetIntegerv(GL_VERTEX_ARRAY_BINDING, &last_vertex_array);
    GLint last_viewport[4]; gl->GetIntegerv(GL_VIEWPORT, last_viewport);
    GLint last_scissor_box[4]; gl->GetIntegerv(GL_SCISSOR_BOX, last_scissor_box);
    GLenum last_blend_src_rgb; gl->GetIntegerv(GL_BLEND_SRC_RGB, (GLint*)&last_blend_src_rgb);
    GLenum last_blend_dst_rgb; gl->GetIntegerv(GL_BLEND_DST_RGB, (GLint*)&last_blend_dst_rgb);
    GLenum last_blend_src_alpha; gl->GetIntegerv(GL_BLEND_SRC_ALPHA, (GLint*)&last_blend_src_alpha);
    GLenum last_blend_dst_alpha; gl->GetIntegerv(GL_BLEND_DST_ALPHA, (GLint*)&last_blend_dst_alpha);
    GLenum last_blend_equation_rgb; gl->GetIntegerv(GL_BLEND_EQUATION_RGB, (GLint*)&last_blend_equation_rgb);
    GLenum last_blend_equation_alpha; gl->GetIntegerv(GL_BLEND_EQUATION_ALPHA, (GLint*)&last_blend_equation_alpha);
    GLboolean last_enable_blend = gl->IsEnabled(GL_BLEND);
    GLboolean last_enable_cull_face = gl->IsEnabled(GL_CULL_FACE);
    GLboolean last_enable_depth_test = gl->IsEnabled(GL_DEPTH_TEST);
    GLboolean last_enable_scissor_test = gl->IsEnabled(GL_SCISSOR_TEST);

    gl->Enable(GL_BLEND);
    gl->BlendEquation(GL_FUNC_ADD);
    gl->BlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    gl->Disable(GL_CULL_FACE);
    gl->Disable(GL_DEPTH_TEST);
    gl->Enable(GL_SCISSOR_TEST);
    gl->Viewport(0, 0, (GLsizei)fb_width, (GLsizei)fb_height);
    float L = draw_data->DisplayPos.x;
    float R = draw_data->DisplayPos.x + draw_data->DisplaySize.x;
    float T = draw_data->DisplayPos.y;
    float B = draw_data->DisplayPos.y + draw_data->DisplaySize.y;
    const float ortho_projection[4][4] =
    {
        { 2.0f/(R-L),   0.0f,         0.0f,   0.0f },
        { 0.0f,         2.0f/(T-B),   0.0f,   0.0f },
        { 0.0f,         0.0f,        -1.0f,   0.0f },
        { (R+L)/(L-R),  (T+B)/(B-T),  0.0f,   1.0f },
    };
    gl->UseProgram(g_ShaderHandle);
    gl->Uniform1i(g_AttribLocationTex, 0);
    gl->UniformMatrix4fv(g_AttribLocationProjMtx, 1, GL_FALSE, &ortho_projection[0][0]);

#define IM_OFFSETOF(_TYPE,_MEMBER)  ((size_t)&(((_TYPE*)0)->_MEMBER))

    GLuint vao_handle = 0;
    gl->GenVertexArrays(1, &vao_handle);
    gl->BindVertexArray(vao_handle);
    gl->BindBuffer(GL_ARRAY_BUFFER, g_VboHandle);
    gl->EnableVertexAttribArray(g_AttribLocationPosition);
    gl->EnableVertexAttribArray(g_AttribLocationUV);
    gl->EnableVertexAttribArray(g_AttribLocationColor);
    gl->VertexAttribPointer(g_AttribLocationPosition, 2, GL_FLOAT, GL_FALSE, sizeof(ImDrawVert), (GLvoid*)IM_OFFSETOF(ImDrawVert, pos));
    gl->VertexAttribPointer(g_AttribLocationUV, 2, GL_FLOAT, GL_FALSE, sizeof(ImDrawVert), (GLvoid*)IM_OFFSETOF(ImDrawVert, uv));
    gl->VertexAttribPointer(g_AttribLocationColor, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(ImDrawVert), (GLvoid*)IM_OFFSETOF(ImDrawVert, col));

    pos = draw_data->DisplayPos;
    for (int n = 0; n < draw_data->CmdListsCount; n++) {
        const ImDrawList *cmd_list = draw_data->CmdLists[n];
        const ImDrawIdx *idx_buffer_offset = 0;

        gl->BindBuffer(GL_ARRAY_BUFFER, g_VboHandle);
        gl->BufferData(GL_ARRAY_BUFFER, (GLsizeiptr)cmd_list->VtxBuffer.Size * sizeof(ImDrawVert), (const GLvoid*)cmd_list->VtxBuffer.Data, GL_STREAM_DRAW);

        gl->BindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_ElementsHandle);
        gl->BufferData(GL_ELEMENT_ARRAY_BUFFER, (GLsizeiptr)cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx), (const GLvoid*)cmd_list->IdxBuffer.Data, GL_STREAM_DRAW);

        for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
            const ImDrawCmd *pcmd = &cmd_list->CmdBuffer.Data[cmd_i];
            ImVec4 clip_rect = { pcmd->ClipRect.x - pos.x, pcmd->ClipRect.y - pos.y, pcmd->ClipRect.z - pos.x, pcmd->ClipRect.w - pos.y };

            if (pcmd->UserCallback)
                continue;
            if (clip_rect.x < fb_width && clip_rect.y < fb_height && clip_rect.z >= 0.0f && clip_rect.w >= 0.0f) {
                // Apply scissor/clipping rectangle
                if (clip_origin_lower_left)
                    gl->Scissor((int)clip_rect.x, (int)(fb_height - clip_rect.w), (int)(clip_rect.z - clip_rect.x), (int)(clip_rect.w - clip_rect.y));
                else
                    gl->Scissor((int)clip_rect.x, (int)clip_rect.y, (int)clip_rect.z, (int)clip_rect.w); // Support for GL 4.5's glClipControl(GL_UPPER_LEFT)

                gl->BindTexture(GL_TEXTURE_2D, (GLuint)(intptr_t)pcmd->TextureId);
                gl->DrawElements(GL_TRIANGLES, (GLsizei)pcmd->ElemCount, sizeof(ImDrawIdx) == 2 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT, idx_buffer_offset);
            }
            idx_buffer_offset += pcmd->ElemCount;
        }
    }
    gl->DeleteVertexArrays(1, &vao_handle);
    gl->UseProgram(last_program);
    gl->BindTexture(GL_TEXTURE_2D, last_texture);
    gl->ActiveTexture(last_active_texture);
    gl->BindVertexArray(last_vertex_array);
    gl->BindBuffer(GL_ARRAY_BUFFER, last_array_buffer);
    gl->BlendEquationSeparate(last_blend_equation_rgb, last_blend_equation_alpha);
    gl->BlendFuncSeparate(last_blend_src_rgb, last_blend_dst_rgb, last_blend_src_alpha, last_blend_dst_alpha);
    if (last_enable_blend) gl->Enable(GL_BLEND); else gl->Disable(GL_BLEND);
    if (last_enable_cull_face) gl->Enable(GL_CULL_FACE); else gl->Disable(GL_CULL_FACE);
    if (last_enable_depth_test) gl->Enable(GL_DEPTH_TEST); else gl->Disable(GL_DEPTH_TEST);
    if (last_enable_scissor_test) gl->Enable(GL_SCISSOR_TEST); else gl->Disable(GL_SCISSOR_TEST);
    gl->Viewport(last_viewport[0], last_viewport[1], (GLsizei)last_viewport[2], (GLsizei)last_viewport[3]);
    gl->Scissor(last_scissor_box[0], last_scissor_box[1], (GLsizei)last_scissor_box[2], (GLsizei)last_scissor_box[3]);
}

// Draw the vertex data (as described by the gl_vao_entry entries) in ptr
// to the screen. num is the number of vertexes. prim is usually GL_TRIANGLES.
// If ptr is NULL, then skip the upload, and use the data uploaded with the
// previous call.
void gl_vao_draw_data(struct gl_vao *vao, GLenum prim, void *ptr, size_t num)
{
    GL *gl = vao->gl;

    if (ptr) {
        gl->BindBuffer(GL_ARRAY_BUFFER, vao->buffer);
        gl->BufferData(GL_ARRAY_BUFFER, num * vao->stride, ptr, GL_STREAM_DRAW);
        gl->BindBuffer(GL_ARRAY_BUFFER, 0);
    }

    gl_vao_bind(vao);

    gl->DrawArrays(prim, 0, num);

    gl_vao_unbind(vao);

    gui_run(vao);
}

static void GLAPIENTRY gl_debug_cb(GLenum source, GLenum type, GLuint id,
                                   GLenum severity, GLsizei length,
                                   const GLchar *message, const void *userParam)
{
    // keep in mind that the debug callback can be asynchronous
    struct mp_log *log = (void *)userParam;
    int level = MSGL_ERR;
    switch (severity) {
    case GL_DEBUG_SEVERITY_NOTIFICATION:level = MSGL_V; break;
    case GL_DEBUG_SEVERITY_LOW:         level = MSGL_INFO; break;
    case GL_DEBUG_SEVERITY_MEDIUM:      level = MSGL_WARN; break;
    case GL_DEBUG_SEVERITY_HIGH:        level = MSGL_ERR; break;
    }
    mp_msg(log, level, "GL: %s\n", message);
}

void gl_set_debug_logger(GL *gl, struct mp_log *log)
{
    if (gl->DebugMessageCallback)
        gl->DebugMessageCallback(log ? gl_debug_cb : NULL, log);
}

// Given a GL combined extension string in extensions, find out whether ext
// is included in it. Basically, a word search.
bool gl_check_extension(const char *extensions, const char *ext)
{
    int len = strlen(ext);
    const char *cur = extensions;
    while (cur) {
        cur = strstr(cur, ext);
        if (!cur)
            break;
        if ((cur == extensions || cur[-1] == ' ') &&
            (cur[len] == '\0' || cur[len] == ' '))
            return true;
        cur += len;
    }
    return false;
}
