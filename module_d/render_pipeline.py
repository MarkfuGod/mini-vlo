import bpy
import os
import math
import json
# ================= 配置区 =================
# 1. 你的动作数据文件夹 (放 .fbx 或 .bvh 文件)
MOTION_DATA_PATH = "C:/Users/lcy/Desktop/HKU final project/motions/" 
# 2. 视频输出的总文件夹
OUTPUT_BASE_DIR = "C:/Users/lcy/Desktop/HKU final project/output_videos/"
# 3. 渲染引擎选择: 'BLENDER_EEVEE'
RENDER_ENGINE = 'BLENDER_EEVEE'

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def setup_render_settings():
    """设置视频输出格式和引擎 (完美适配 Blender 5.1)"""
    scene = bpy.context.scene
    scene.render.engine = RENDER_ENGINE
    
    # === 核心修改点：应用你抓取到的 5.1 版本 API ===
    scene.render.image_settings.file_format = 'FFMPEG'
    
    # 兼容 5.1 的新媒体类型属性
    try:
        scene.render.image_settings.media_type = 'VIDEO'
    except AttributeError:
        pass 

    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    
    # 分辨率
    #scene.render.resolution_x = 512
    #scene.render.resolution_y = 512
    # 1. 提高分辨率 (例如改为 1024x1024 或 1920x1080)
    scene.render.resolution_x = 1024 
    scene.render.resolution_y = 1024
    
    # 2. 提升画面百分比 (确保是 100%)
    scene.render.resolution_percentage = 100

    # 3. 提高 EEVEE 引擎的抗锯齿采样率 (让边缘更平滑，画质更清晰)
    scene.eevee.taa_render_samples = 64

def clear_imported_objects():
    """清理上一轮导入的骨架和物体，保持场景上下文"""
    for obj in bpy.data.objects:
        # === 修改点 1：把 '苏珊娜' 加入不删除的白名单 ===
        if obj.type in ['ARMATURE', 'MESH'] and obj.name not in ['平面', '苏珊娜']: 
            bpy.data.objects.remove(obj, do_unlink=True)

def add_scene_context(action_name, armature, scene):
    """
    根据动作名称中的关键字，动态生成简单的场景几何体（支持骨骼动态双向追踪）
    """
    name_lower = action_name.lower()
    
    # 获取臀部骨骼（通用逻辑，跨越和坐下都需要用到）
    hips_bone = None
    if armature:
        hips_keywords = ["Hips", "hips", "Pelvis", "pelvis", "Root", "root"]
        hips_bone = next((pb for pb in armature.pose.bones if any(k in pb.name for k in hips_keywords)), None)

    # ================= 1. 跨越、跳跃动作 -> 追踪【最高点】生成障碍物 =================
    if "vault" in name_lower or "step" in name_lower:
        if hips_bone:
            highest_z = float('-inf')
            best_location = None
            
            current_frame = scene.frame_current
            
            # 遍历所有帧，寻找腾空最高点
            for f in range(scene.frame_start, scene.frame_end + 1):
                scene.frame_set(f)
                bpy.context.view_layer.update() # 强制刷新矩阵
                
                world_loc = armature.matrix_world @ hips_bone.matrix.translation
                
                if world_loc.z > highest_z:
                    highest_z = world_loc.z
                    # 障碍物应该在最高点的正下方地面上
                    # 因为立方体Z轴缩放了0.5，所以高度是0.5米，Z坐标设为0.25正好贴紧地面
                    best_location = (world_loc.x, world_loc.y, 0.25)
            
            scene.frame_set(current_frame) # 恢复到初始帧
            
            if best_location:
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=best_location)
                obj = bpy.context.active_object
                obj.name = "Context_Obstacle"
                obj.scale = (2.0, 0.3, 0.5) 
                print(f"  -> [🔥 骨骼追踪成功] 已在跨越最高点坐标 {best_location} 生成障碍物。")
            else:
                # 容错：如果计算出错，用默认坐标
                bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.02, 0.25))
                obj = bpy.context.active_object
                obj.name = "Context_Obstacle"
                obj.scale = (2.0, 0.3, 0.5)
        else:
            print(f"⚠️ 未找到有效骨架，执行默认障碍物生成。")
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, -0.02, 0.25))
            obj = bpy.context.active_object
            obj.name = "Context_Obstacle"
            obj.scale = (2.0, 0.3, 0.5)

    # ================= 2. 坐下动作 -> 追踪【最低点】生成椅子 =================
    elif "sit" in name_lower:
        if hips_bone:
            lowest_z = float('inf')
            best_location = None
            
            current_frame = scene.frame_current
            
            # 遍历所有帧，寻找臀部最低点
            for f in range(scene.frame_start, scene.frame_end + 1):
                scene.frame_set(f)
                bpy.context.view_layer.update() 
                
                world_loc = armature.matrix_world @ hips_bone.matrix.translation
                
                if world_loc.z < lowest_z:
                    lowest_z = world_loc.z
                    # 椅子在屁股正下方，Z坐标设为0.25贴紧地面
                    best_location = (world_loc.x, world_loc.y, 0.25)
            
            scene.frame_set(current_frame)
            
            if best_location:
                bpy.ops.mesh.primitive_cube_add(size=0.5, location=best_location)
                obj = bpy.context.active_object
                obj.name = "Context_Chair"
                print(f"  -> [🔥 骨骼追踪成功] 已在屁股最低点坐标 {best_location} 生成椅子。")
            else:
                bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0.5, 0.25))
                obj = bpy.context.active_object
                obj.name = "Context_Chair"
        else:
            print(f"⚠️ 未找到有效骨架，使用默认坐标生成椅子。")
            bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0.5, 0.25))
            obj = bpy.context.active_object
            obj.name = "Context_Chair"
            
            
   # ================= 3. 向下跳跃/跌落动作 -> 追踪【跳跃方向】并在【边缘】生成高台 =================
    elif "drop" in name_lower or "jump_down" in name_lower:
        if hips_bone:
            # 1. 获取起跳点 (第 1 帧)
            scene.frame_set(scene.frame_start)
            bpy.context.view_layer.update()
            start_loc = armature.matrix_world @ hips_bone.matrix.translation.copy()
            
            # 2. 获取落地结束点 (最后 1 帧)
            scene.frame_set(scene.frame_end)
            bpy.context.view_layer.update()
            end_loc = armature.matrix_world @ hips_bone.matrix.translation.copy()
            
            # 3. 计算水平跳跃的“方向向量”
            dx = end_loc.x - start_loc.x
            dy = end_loc.y - start_loc.y
            dist = math.sqrt(dx**2 + dy**2)
            
            if dist > 0.1:
                dir_x = dx / dist
                dir_y = dy / dist
            else:
                dir_x, dir_y = 0, -1 # 容错：如果没有明显位移，默认朝 -Y 方向
                
            # 恢复到第 1 帧，准备寻找脚底高度
            scene.frame_set(scene.frame_start)
            bpy.context.view_layer.update()
        else:
            start_loc = None
            dir_x, dir_y = 0, 0

        # 4. 寻找脚底绝对最低点 (精确贴合脚底)
        foot_keywords = ["foot", "toe", "heel", "ankle"]
        foot_bones = [pb for pb in armature.pose.bones if any(k in pb.name.lower() for k in foot_keywords)]
        platform_z_surface = float('inf')
        
        if foot_bones:
            for fb in foot_bones:
                loc = armature.matrix_world @ fb.matrix.translation
                if loc.z < platform_z_surface:
                    platform_z_surface = loc.z
        else:
            for pb in armature.pose.bones:
                loc = armature.matrix_world @ pb.matrix.translation
                if loc.z < platform_z_surface:
                    platform_z_surface = loc.z

        # 5. 【核心修复】：动态偏移高台中心，让角色站在边缘
        if platform_z_surface > 0.05 and start_loc:
            platform_size = 2.0 # 平台长宽(2x2米)
            
            # 将平台中心往“跳跃反方向”拉。
            # 半径是 1.0 米，拉回 0.8 米，只留 20cm 在角色身前（刚好够放脚尖，防止悬空）
            offset_dist = (platform_size / 2) - 0.2  
            
            center_x = start_loc.x - (dir_x * offset_dist)
            center_y = start_loc.y - (dir_y * offset_dist)

            # 生成高台
            bpy.ops.mesh.primitive_cube_add(
                size=1.0, 
                location=(center_x, center_y, platform_z_surface / 2)
            )
            obj = bpy.context.active_object
            obj.name = "Context_Platform"
            obj.scale = (platform_size, platform_size, platform_z_surface) 
            
            print(f"  -> [🔥 骨骼追踪成功] 角色朝({dir_x:.2f}, {dir_y:.2f})跳跃。已将高台后移，角色现处于悬崖边缘！")
        else:
            print("⚠️ 起跳高度过低，不足以生成高台。")
    
    
def export_trajectory_to_json(armature, scene, video_folder, file_name):
    """
    遍历时间轴，提取骨骼 3D 轨迹并保存为 JSON
    """
    trajectory_data = {}
    
    # 定义你要提取的部位，以及它们在不同动捕标准下可能出现的名字
    target_bones_keywords = {
        "Root": ["Hips", "hips", "Root", "root", "Pelvis", "pelvis"],
        "Hand_R": ["RightHand", "Hand_R", "hand.R", "hand_r"],
        "Hand_L": ["LeftHand", "Hand_L", "hand.L", "hand_l"]
    }
    
    # 动态匹配当前骨架里实际的骨骼名称
    actual_bones = {}
    for target_key, keywords in target_bones_keywords.items():
        for pb in armature.pose.bones:
            if any(keyword in pb.name for keyword in keywords):
                actual_bones[target_key] = pb.name
                break
                
    # 如果找不到根节点，直接返回
    if "Root" not in actual_bones:
        print(f"⚠️ 未能在 {file_name} 中找到 Root 骨骼，跳过轨迹导出。")
        return

    print(f"开始提取 {file_name} 的 3D 轨迹数据...")
    
    # 遍历动画的每一帧
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        # === 核心：必须强制刷新视图层，否则骨骼矩阵不会随动画更新 ===
        bpy.context.view_layer.update() 
        
        frame_data = {}
        for target_key, bone_name in actual_bones.items():
            pose_bone = armature.pose.bones[bone_name]
            # 计算骨骼在世界坐标系下的真实物理位置 (骨架矩阵 @ 骨骼矩阵)
            world_loc = armature.matrix_world @ pose_bone.matrix.translation
            
            # 保留 4 位小数，使 JSON 文件不会过于庞大
            frame_data[target_key] = {
                "x": round(world_loc.x, 4), 
                "y": round(world_loc.y, 4), 
                "z": round(world_loc.z, 4)
            }
            
        trajectory_data[f"frame_{frame}"] = frame_data

    # 写入 JSON 文件到对应的视频文件夹下
    json_path = os.path.join(video_folder, f"{file_name}_trajectory.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(trajectory_data, f, indent=4)
        
    print(f"✅ 已成功导出 3D 轨迹至: {json_path}")
    
    
def render_action_video(action_file):
    """处理单个动作文件的渲染逻辑"""
    file_name = os.path.basename(action_file).split('.')[0]
    video_folder = os.path.join(OUTPUT_BASE_DIR, file_name)
    os.makedirs(video_folder, exist_ok=True)

    # 1. 导入动作 (加入容错机制)
    try:
        if action_file.endswith('.fbx'):
            bpy.ops.import_scene.fbx(filepath=action_file)
        elif action_file.endswith('.bvh'):
            bpy.ops.import_anim.bvh(filepath=action_file)
            
    except RuntimeError as e:
        print(f"⚠️ 跳过文件 {action_file}: 导入失败，可能文件已损坏。错误信息: {e}")
        return 

    # 2. 自动设置动画时长
    scene = bpy.context.scene
    armature = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
    if armature and armature.animation_data:
        action = armature.animation_data.action
        scene.frame_start = int(action.frame_range[0])
        scene.frame_end = int(action.frame_range[1])
        
        # ==================== 核心修改点 ====================
        # 必须在导入动作、识别出 armature 之后再调用生成场景上下文
        # 并且传入 armature 和 scene 参数
        add_scene_context(file_name, armature, scene)
        # ====================================================
        
        # 导出 3D 轨迹 (如果你之前加了这个功能的话)
        export_trajectory_to_json(armature, scene, video_folder, file_name)
  

    # ================= 自动化绑定第一人称相机 =================
    cam_ego = bpy.data.objects.get("摄像机")
    # ... 后续代码保持不变 ...

    if cam_ego and armature:
        head_bone_name = None
        for bone in armature.data.bones:
            if "Head" in bone.name or "head" in bone.name:
                head_bone_name = bone.name
                break
        
        if head_bone_name:
            cam_ego.parent = armature
            cam_ego.parent_type = 'BONE'
            cam_ego.parent_bone = head_bone_name
            
            cam_ego.location = (0, 0, 2) 
            
            rot_x = math.radians(180)  
            rot_y = math.radians(0)    
            rot_z = math.radians(180)  
            
            cam_ego.rotation_euler = (rot_x, rot_y, rot_z) 
            
            print(f"成功将摄像机绑定到新骨架的 {head_bone_name} 骨骼上！视角已修正。")

    # 3. 渲染多视角 
    cameras = {
        "Fixed_View": bpy.data.objects.get("Camera"),
        "Ego_View": bpy.data.objects.get("摄像机")
    }

    for view_name, cam in cameras.items():
        if cam:
            scene.camera = cam
            # === 核心修改点：改回 .mp4 后缀 ===
            scene.render.filepath = os.path.join(video_folder, f"{view_name}.mp4")
            print(f"正在渲染 {file_name} 的 {view_name} 视角...")
            bpy.ops.render.render(animation=True)

# ================= 主执行逻辑 =================
setup_render_settings()

for file in os.listdir(MOTION_DATA_PATH):
    if file.endswith(('.fbx', '.bvh')):
        clear_imported_objects() 
        render_action_video(os.path.join(MOTION_DATA_PATH, file))
        
    

print("所有动作视频渲染任务已完成！")